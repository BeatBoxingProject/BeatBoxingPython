import cv2
import numpy as np
import time
import os

# ====================================================================
# --- 1. CONFIGURATION - YOU MUST CHANGE THIS ---
# ====================================================================

# --- Camera Stream URLs ---
url_1 = "http://192.168.137.102/stream?framesize=13"  # Left Camera
url_2 = "http://192.168.137.101/stream?framesize=13"  # Right Camera

# --- Chessboard Configuration ---
# Number of inner corners (horizontal, vertical)
CHECKERBOARD = (9, 6)  # 8x6 corners. A 9x7 square board.
# Physical size of a square in your chosen unit (e.g., 25mm, 0.025m)
SQUARE_SIZE_METERS = 0.025

# --- Calibration Settings ---
# How many good image pairs to capture before calibrating
IMAGES_NEEDED = 25

# ====================================================================
# --- 2. HELPERS & SETUP ---
# ====================================================================

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D "object points" (e.g., (0,0,0), (1,0,0), ... (7,5,0))
# These are the "real world" coordinates of the corners.
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_METERS  # Scale to real-world size

# Arrays to store points from all good images
objpoints = []  # 3D points in real-world space
imgpoints_l = []  # 2D points in left camera plane
imgpoints_r = []  # 2D points in right camera plane


# Function to safely connect to a camera (from our previous script)
def connect_to_camera(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Could not open stream at {url}")
        return None
    return cap


def save_calibration_data(filename, matrices):
    """Saves all calibration matrices to a .yml file."""
    print(f"Saving calibration data to {filename}...")

    # Use cv2.FileStorage to write the data
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
        print("Failed to connect to one or both cameras. Exiting.")
        return

    print("Connections successful. Starting loop...")
    print("\n--- INSTRUCTIONS ---")
    print("Hold your chessboard in front of both cameras.")
    print("Press [Spacebar] to capture a good image pair.")
    print(f"Press [c] to calibrate (after {IMAGES_NEEDED} images).")
    print("Press [q] to quit.")

    img_count = 0
    frame_shape = None  # To store frame size for calibration

    while True:
        try:
            # --- Handle Reconnection ---
            if not cap_l.isOpened():
                print("Left camera disconnected. Reconnecting...")
                cap_l = connect_to_camera(url_1)
                time.sleep(3)
                continue
            if not cap_r.isOpened():
                print("Right camera disconnected. Reconnecting...")
                cap_r = connect_to_camera(url_2)
                time.sleep(3)
                continue

            # --- Read Frames ---
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l or not ret_r:
                print("Error: Failed to grab frame from one or both cameras.")
                cap_l.release()
                cap_r.release()
                time.sleep(1)
                continue

            if frame_shape is None:
                frame_shape = frame_l.shape[:2][::-1]  # (width, height)

            # --- User Interface ---
            # Create a display copy
            display_l = frame_l.copy()
            display_r = frame_r.copy()

            # Add text
            cv2.putText(display_l, f"Images: {img_count}/{IMAGES_NEEDED}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(display_l, "Press 'Space' to capture", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(display_l, "Press 'c' to calibrate", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(display_l, "Press 'q' to quit", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- Keyboard Input ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting...")
                break

            elif key == ord(' '):
                # --- Capture Image Pair ---
                print(f"Attempting capture {img_count + 1}...")
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                # Find chessboard corners
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

                # If found in BOTH images, save them
                if ret_l and ret_r:
                    img_count += 1
                    print(f"Success! Found corners in both images. ({img_count}/{IMAGES_NEEDED})")

                    # Refine corners
                    corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                    corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

                    # Add points to our lists
                    objpoints.append(objp)
                    imgpoints_l.append(corners2_l)
                    imgpoints_r.append(corners2_r)

                    # Draw corners on display
                    cv2.drawChessboardCorners(display_l, CHECKERBOARD, corners2_l, ret_l)
                    cv2.drawChessboardCorners(display_r, CHECKERBOARD, corners2_r, ret_r)
                else:
                    print("Error: Corners not found in one or both images. Try a different angle.")

            elif key == ord('c'):
                # --- Calibrate ---
                if img_count < IMAGES_NEEDED:
                    print(f"Error: Need at least {IMAGES_NEEDED} images. Only have {img_count}.")
                else:
                    print(f"Starting calibration with {img_count} images. This may take a minute...")

                    # --- Individual Camera Calibration (Optional but Recommended) ---
                    # We first calibrate each camera individually
                    print("Calibrating left camera...")
                    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, frame_shape,
                                                                                 None, None)
                    print("Calibrating right camera...")
                    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, frame_shape,
                                                                                 None, None)

                    # --- Stereo Calibration ---
                    print("Running stereo calibration...")

                    # Set flags: CALIB_FIX_INTRINSIC = cameras are pre-calibrated
                    flags = cv2.CALIB_FIX_INTRINSIC

                    (ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F) = cv2.stereoCalibrate(
                        objpoints, imgpoints_l, imgpoints_r,
                        mtx_l, dist_l,
                        mtx_r, dist_r,
                        frame_shape,
                        flags=flags,
                        criteria=criteria
                    )

                    print(f"Stereo calibration 'ret' value: {ret_stereo}")
                    print("Calibration complete. Calculating rectification...")

                    # --- Stereo Rectification ---
                    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                        mtx_l, dist_l,
                        mtx_r, dist_r,
                        frame_shape, R, T,
                        flags=cv2.CALIB_ZERO_DISPARITY,
                        alpha=-1  # -1 = auto-scaling
                    )

                    print("Rectification complete. Saving data...")

                    # --- Save to File ---
                    calibration_data = {
                        'cameraMatrix1': mtx_l,
                        'distCoeffs1': dist_l,
                        'cameraMatrix2': mtx_r,
                        'distCoeffs2': dist_r,
                        'R': R,
                        'T': T,
                        'R1': R1,
                        'R2': R2,
                        'P1': P1,
                        'P2': P2,
                        'Q': Q,
                        'frame_width': frame_shape[0],
                        'frame_height': frame_shape[1]
                    }

                    save_calibration_data("stereo_calib.yml", calibration_data)
                    print("\nCalibration successful! File 'stereo_calib.yml' saved.")
                    print("You can now press 'q' to quit.")

            # Show the live frames
            cv2.imshow('Left Camera', display_l)
            cv2.imshow('Right Camera', display_r)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Resetting connections...")
            cap_l.release()
            cap_r.release()
            time.sleep(3)

    # Clean up
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()