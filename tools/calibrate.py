import sys
import os
# Allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
# Import the path from config to ensure we save to the right place
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

def connect_to_camera(url):
    print(f"Connecting to {url}...")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Could not open stream at {url}")
        return None
    return cap

def save_calibration_data(filepath, matrices):
    # Ensure the 'data' directory exists before writing
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

    print(f"Saving calibration data to {filepath}...")
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print(f"Error: Could not open {filepath} for writing.")
        return
    for key, value in matrices.items():
        fs.write(key, value)
    fs.release()
    print(f"Successfully saved data.")

def main():
    cap_l = connect_to_camera(LEFT_CAM_URL_CALIB)
    cap_r = connect_to_camera(RIGHT_CAM_URL_CALIB)

    if cap_l is None or cap_r is None:
        print("Failed to connect. Exiting.")
        return

    # Setup display windows
    cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Left Camera', 640, 480)
    cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Right Camera', 640, 480)

    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_METERS

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    
    img_count = 0
    frame_shape = None
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    print("\n--- INSTRUCTIONS ---")
    print("Hold checkerboard. Press [Space] to capture, [c] to calibrate, [q] to quit.")

    while True:
        try:
            # Reconnect logic for blocking capture
            if not cap_l.isOpened(): cap_l = connect_to_camera(LEFT_CAM_URL_CALIB)
            if not cap_r.isOpened(): cap_r = connect_to_camera(RIGHT_CAM_URL_CALIB)

            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l or not ret_r: continue

            if frame_shape is None:
                frame_shape = frame_l.shape[:2][::-1]

            display_l = frame_l.copy()
            display_r = frame_r.copy()

            cv2.putText(display_l, f"Images: {img_count}/{IMAGES_NEEDED}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                print(f"Attempting capture {img_count + 1}...")
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                ret_l_found, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, flags=find_flags)
                ret_r_found, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, flags=find_flags)

                if ret_l_found: cv2.drawChessboardCorners(display_l, CHECKERBOARD, corners_l, ret_l_found)
                if ret_r_found: cv2.drawChessboardCorners(display_r, CHECKERBOARD, corners_r, ret_r_found)

                if ret_l_found and ret_r_found:
                    img_count += 1
                    print(f"Success! ({img_count}/{IMAGES_NEEDED})")
                    corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
                    corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)
                    objpoints.append(objp)
                    imgpoints_l.append(corners2_l)
                    imgpoints_r.append(corners2_r)
                else:
                    print("Corners not found in both images.")

            elif key == ord('c'):
                if img_count < IMAGES_NEEDED:
                    print(f"Need {IMAGES_NEEDED} images.")
                else:
                    print("Starting calibration (this may take a moment)...")
                    
                    # Individual Calibration
                    print("Calibrating Left...")
                    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
                        objpoints, imgpoints_l, frame_shape, None, None)
                    print(f"  >> Left RMS: {ret_l:.4f}")

                    print("Calibrating Right...")
                    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
                        objpoints, imgpoints_r, frame_shape, None, None)
                    print(f"  >> Right RMS: {ret_r:.4f}")

                    # Stereo Calibration
                    print("Stereo Calibrating...")
                    # We LOCK the principal point and IGNORE tangential distortion to prevent "explosion"
                    flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH

                    (ret_s, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F) = cv2.stereoCalibrate(
                        objpoints, imgpoints_l, imgpoints_r,
                        mtx_l, dist_l, mtx_r, dist_r,
                        frame_shape, flags=flags, criteria=CRITERIA
                    )

                    print(f"RMS Error: {ret_s}")
                    if ret_s > 1.0:
                        print("⚠️ WARNING: RMS Error is too high! (> 1.0). Results will be warped.")
                        print("Try capturing more images or holding the board steadier.")
                    else:
                        print(f"✅ Calibration looks good! (RMS: {ret_s:.4f})")

                    # Rectification
                    # alpha=-1 (Auto): Tries to find a "nice" balance. If your distortion coefficients are high (common with cheap lenses), it might mistakenly decide that the "nice" area is a tiny 10x10 pixel region in the center, stretching it to fill the whole screen.
                    # alpha=0 (Crop): "Zoom in until the image fits." This is usually what you want for gaming, but if your alignment is bad, you might lose too much of the image.
                    # alpha=1 (Full): "Zoom out so I see everything." This results in curved black borders around the image (like a fisheye effect), but it guarantees you will actually see the video feed.
                    print("Rectifying...")
                    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                        mtx_l, dist_l, mtx_r, dist_r,
                        frame_shape, R, T,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
                    )

                    data = {
                        'cameraMatrix1': mtx_l, 'distCoeffs1': dist_l,
                        'cameraMatrix2': mtx_r, 'distCoeffs2': dist_r,
                        'R': R, 'T': T, 'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
                        'frame_width': frame_shape[0], 'frame_height': frame_shape[1]
                    }
                    
                    # USE THE CONFIG PATH HERE
                    save_calibration_data(CALIB_FILE, data)
                    print("Done! Press 'q' to quit.")

            cv2.imshow('Left Camera', display_l)
            cv2.imshow('Right Camera', display_r)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()