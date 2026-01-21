import sys
import os
import cv2
import numpy as np
import time

# Allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.camera import CameraStream
from core.config import LEFT_CAM_URL_CALIB, RIGHT_CAM_URL_CALIB, CALIB_FILE

# ====================================================================
# --- CONFIGURATION ---
# ====================================================================
CHECKERBOARD = (9, 6)
SQUARE_SIZE_METERS = 0.025
IMAGES_NEEDED = 15
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ====================================================================
# --- HELPERS ---
# ====================================================================

def save_calibration_data(filepath, matrices):
    """Saves the calibration results to a YAML file."""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Saving calibration data to {filepath}...")
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print(f"Error: Could not open {filepath} for writing.")
        return
        
    for key, value in matrices.items():
        fs.write(key, value)
    fs.release()
    print("Successfully saved data.")

def check_and_reconnect(cam_obj, url, name):
    """
    Checks if a camera stream is active. If not, attempts to restart it.
    Returns the active (or new) CameraStream object.
    """
    if cam_obj is None or not cam_obj.ret:
        print(f"⚠️  {name} Camera lost/invalid. Reconnecting to {url}...")
        
        # Clean up old thread if it exists
        if cam_obj is not None:
            cam_obj.stop()
        
        # specific sleep to prevent rapid-fire reconnection spam
        time.sleep(1.0) 
        
        # Attempt new connection
        try:
            new_cam = CameraStream(url, name)
            # Wait briefly to see if it grabs a frame
            time.sleep(1.0)
            if new_cam.ret:
                print(f"✅ {name} Camera Connected!")
                return new_cam
            else:
                new_cam.stop()
                return None
        except Exception as e:
            print(f"❌ Error connecting to {name}: {e}")
            return None

    return cam_obj

def draw_status(img, text, color=(0, 0, 255)):
    """Overlay status text on a black image for disconnected states."""
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    h, w = img.shape[:2]
    cv2.putText(img, text, (50, h // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

# ====================================================================
# --- MAIN ENGINE ---
# ====================================================================

def main():
    # 1. Start Cameras
    print("Starting Camera Streams...")
    cam_l = CameraStream(LEFT_CAM_URL_CALIB, "Left")
    cam_r = CameraStream(RIGHT_CAM_URL_CALIB, "Right")

    # Give them a moment to warm up
    time.sleep(2.0)

    # Setup Windows
    cv2.namedWindow('Stereo Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo Calibration', 1280, 480)

    # Calibration Storage
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_METERS

    objpoints = []     # 3d point in real world space
    imgpoints_l = []   # 2d points in image plane.
    imgpoints_r = []   # 2d points in image plane.
    
    img_count = 0
    frame_shape = None
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    print("\n--- INSTRUCTIONS ---")
    print("1. Hold checkerboard visible to BOTH cameras.")
    print("2. Press [Space] to capture a pair.")
    print("3. Press [c] to calculate calibration.")
    print("4. Press [q] to quit.")

    while True:
        # A. Connection Health Check
        cam_l = check_and_reconnect(cam_l, LEFT_CAM_URL_CALIB, "Left")
        cam_r = check_and_reconnect(cam_r, RIGHT_CAM_URL_CALIB, "Right")

        # B. Frame Retrieval
        ret_l, frame_l = cam_l.read() if cam_l else (False, None)
        ret_r, frame_r = cam_r.read() if cam_r else (False, None)

        # C. Visualization Handling
        display_l = frame_l.copy() if ret_l else None
        display_r = frame_r.copy() if ret_r else None

        if display_l is None:
            display_l = draw_status(None, "Waiting for Left...", (0, 0, 255))
        if display_r is None:
            display_r = draw_status(None, "Waiting for Right...", (0, 0, 255))

        # Ensure consistent size for stacking
        if display_l.shape != display_r.shape:
             display_r = cv2.resize(display_r, (display_l.shape[1], display_l.shape[0]))

        # Combine for a single cleaner window
        combined_view = np.hstack((display_l, display_r))
        
        # Overlay Status
        status_color = (0, 255, 0) if (ret_l and ret_r) else (0, 0, 255)
        status_text = f"Images: {img_count}/{IMAGES_NEEDED}"
        cv2.putText(combined_view, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        cv2.imshow('Stereo Calibration', combined_view)
        
        # D. Input Handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Only allow actions if both cameras are online
        if ret_l and ret_r:
            if frame_shape is None:
                frame_shape = frame_l.shape[:2][::-1]

            if key == ord(' '):
                print(f"Attempting capture {img_count + 1}...")
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                found_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, flags=find_flags)
                found_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, flags=find_flags)

                if found_l and found_r:
                    # Refine corners
                    corners_l_opt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
                    corners_r_opt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)

                    # Draw success on current frame for feedback
                    cv2.drawChessboardCorners(display_l, CHECKERBOARD, corners_l_opt, found_l)
                    cv2.drawChessboardCorners(display_r, CHECKERBOARD, corners_r_opt, found_r)
                    
                    # Update visualized frame immediately
                    combined_debug = np.hstack((display_l, display_r))
                    cv2.imshow('Stereo Calibration', combined_debug)
                    cv2.waitKey(500) # Pause briefly to show success

                    objpoints.append(objp)
                    imgpoints_l.append(corners_l_opt)
                    imgpoints_r.append(corners_r_opt)
                    img_count += 1
                    print(f"✅ Capture Success! ({img_count}/{IMAGES_NEEDED})")
                else:
                    print(f"⚠️  Corners not found. L:{found_l} R:{found_r}")

            elif key == ord('c'):
                if img_count < IMAGES_NEEDED:
                    print(f"⚠️  Not enough images. Need {IMAGES_NEEDED}, have {img_count}.")
                else:
                    print("\n--- STARTING CALIBRATION ---")
                    
                    print("1. Calibrating Left Camera...")
                    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
                        objpoints, imgpoints_l, frame_shape, None, None)

                    print("2. Calibrating Right Camera...")
                    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
                        objpoints, imgpoints_r, frame_shape, None, None)

                    print("3. performing Stereo Calibration...")
                    # CHANGE: Allow the optimizer to adjust intrinsics for a better stereo fit
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS
                    (ret_s, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F) = cv2.stereoCalibrate(
                        objpoints, imgpoints_l, imgpoints_r,
                        mtx_l, dist_l, mtx_r, dist_r,
                        frame_shape, flags=flags, criteria=CRITERIA
                    )

                    print("4. Calculating Rectification Transforms...")
                    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                        mtx_l, dist_l, mtx_r, dist_r,
                        frame_shape, R, T,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
                    )

                    data = {
                        'cameraMatrix1': mtx_l, 'distCoeffs1': dist_l,
                        'cameraMatrix2': mtx_r, 'distCoeffs2': dist_r,
                        'R': R, 'T': T, 'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
                        'frame_width': frame_shape[0], 'frame_height': frame_shape[1]
                    }
                    
                    save_calibration_data(CALIB_FILE, data)
                    print("\n🎉 Calibration Complete! You can now run 'main.py'.")
                    break

    # Cleanup
    if cam_l: cam_l.stop()
    if cam_r: cam_r.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()