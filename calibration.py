import cv2
import numpy as np
import time

# ====================================================================
# --- 1. CONFIGURATION ---
# ====================================================================

# Use Framesize 13 (UXGA - 1600x1200) for maximum precision
url_1 = "http://192.168.137.101/stream?framesize=13"
url_2 = "http://192.168.137.102/stream?framesize=13"

# Check your board! You had (8, 5) in previous attempts.
# Ensure this matches the INNER corners.
CHECKERBOARD = (9, 6)
SQUARE_SIZE_METERS = 0.025
IMAGES_NEEDED = 15

# ====================================================================
# --- 2. HELPERS & SETUP ---
# ====================================================================

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_METERS

objpoints = []
imgpoints_l = []
imgpoints_r = []


def connect_to_camera(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Could not open stream at {url}")
        return None
    return cap


def save_calibration_data(filename, matrices):
    print(f"Saving calibration data to {filename}...")
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print(f"Error: Could not open {filename} for writing.")
        return
    for key, value in matrices.items():
        fs.write(key, value)
    fs.release()
    print(f"Successfully saved data.")


# ====================================================================
# --- 3. MAIN CALIBRATION LOOP ---
# ====================================================================

def main():
    print("Connecting to cameras...")
    cap_l = connect_to_camera(url_1)
    cap_r = connect_to_camera(url_2)

    if cap_l is None or cap_r is None:
        print("Failed to connect. Exiting.")
        return

    print("Connections successful.")

    # --- NEW: SETUP RESIZABLE WINDOWS ---
    # This allows us to see the 1600x1200 stream on a smaller screen
    # without actually resizing the image data (which would ruin calibration)
    cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Left Camera', 640, 480)  # Display size

    cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Right Camera', 640, 480)  # Display size
    # ------------------------------------

    print("\n--- INSTRUCTIONS ---")
    print("Hold your chessboard in front of both cameras.")
    print("Press [Spacebar] to capture a good image pair.")
    print(f"Press [c] to calibrate (after {IMAGES_NEEDED} images).")
    print("Press [q] to quit.")

    img_count = 0
    frame_shape = None

    # Define flags for better corner finding
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    while True:
        try:
            if not cap_l.isOpened():
                cap_l = connect_to_camera(url_1)
                time.sleep(2)
                continue
            if not cap_r.isOpened():
                cap_r = connect_to_camera(url_2)
                time.sleep(2)
                continue

            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l or not ret_r:
                # print("Waiting for frames...")
                continue

            if frame_shape is None:
                frame_shape = frame_l.shape[:2][::-1]

            display_l = frame_l.copy()
            display_r = frame_r.copy()

            cv2.putText(display_l, f"Images: {img_count}/{IMAGES_NEEDED}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0), 4)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                print(f"Attempting capture {img_count + 1}...")
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                # Find corners using the high-res image
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, flags=find_flags)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, flags=find_flags)

                # Draw corners on the display copy (visual only)
                if ret_l: cv2.drawChessboardCorners(display_l, CHECKERBOARD, corners_l, ret_l)
                if ret_r: cv2.drawChessboardCorners(display_r, CHECKERBOARD, corners_r, ret_r)

                if ret_l and ret_r:
                    img_count += 1
                    print(f"Success! ({img_count}/{IMAGES_NEEDED})")

                    corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                    corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

                    objpoints.append(objp)
                    imgpoints_l.append(corners2_l)
                    imgpoints_r.append(corners2_r)
                else:
                    print("Corners not found in both images.")

            elif key == ord('c'):
                if img_count < IMAGES_NEEDED:
                    print(f"Need {IMAGES_NEEDED} images.")
                else:
                    print("Starting calibration...")

                    # Use named arguments to avoid linter warnings
                    print("Calibrating Left...")
                    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
                        objpoints, imgpoints_l, frame_shape, cameraMatrix=None, distCoeffs=None)

                    print("Calibrating Right...")
                    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
                        objpoints, imgpoints_r, frame_shape, cameraMatrix=None, distCoeffs=None)

                    print("Stereo Calibrating...")
                    flags = cv2.CALIB_FIX_INTRINSIC
                    (ret_s, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F) = cv2.stereoCalibrate(
                        objectPoints=objpoints, imagePoints1=imgpoints_l, imagePoints2=imgpoints_r,
                        cameraMatrix1=mtx_l, distCoeffs1=dist_l,
                        cameraMatrix2=mtx_r, distCoeffs2=dist_r,
                        imageSize=frame_shape, flags=flags, criteria=criteria
                    )

                    print("Rectifying...")
                    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                        cameraMatrix1=mtx_l, distCoeffs1=dist_l,
                        cameraMatrix2=mtx_r, distCoeffs2=dist_r,
                        imageSize=frame_shape, R=R, T=T,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
                    )

                    data = {
                        'cameraMatrix1': mtx_l, 'distCoeffs1': dist_l,
                        'cameraMatrix2': mtx_r, 'distCoeffs2': dist_r,
                        'R': R, 'T': T, 'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
                        'frame_width': frame_shape[0], 'frame_height': frame_shape[1]
                    }
                    save_calibration_data("stereo_calib.yml", data)
                    print("Done! Press 'q' to quit.")

            # Show the resized windows
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