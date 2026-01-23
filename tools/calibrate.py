import sys
import os
import glob
import cv2
import numpy as np

# Allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import CALIB_FILE, DATA_DIR

# ====================================================================
# --- CONFIGURATION ---
# ====================================================================
CHECKERBOARD = (9, 6)  # Inner corners (rows, cols)
SQUARE_SIZE_METERS = 0.025  # Size of one square in meters
IMAGE_FOLDER = os.path.join(DATA_DIR, "calibration_images")

# --- CAMERA ORDER SETTING ---
# Set to True if "left_xx.jpg" is actually the RIGHT camera view
SWAP_CAMERAS = True

# Flags for ESP32/Cheap Lenses
STEREO_FLAGS = (
        cv2.CALIB_FIX_PRINCIPAL_POINT |
        cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_ZERO_TANGENT_DIST |
        cv2.CALIB_SAME_FOCAL_LENGTH
)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


# ====================================================================
# --- MAIN SCRIPT ---
# ====================================================================

def save_calibration_data(filepath, matrices):
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
    print(f"Successfully saved data.")


def main():
    print(f"--- STEREO CALIBRATION FROM FILES ---")
    print(f"Searching in: {IMAGE_FOLDER}")
    if SWAP_CAMERAS:
        print("⚠️ MODE: SWAPPING CAMERAS (Treating 'left' files as Right Camera)")

    # 1. Find images
    left_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "left_*.jpg")))

    if not left_images:
        print("No images found! Run 'tools/capture_images.py' first.")
        return

    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_METERS

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    frame_shape = None
    valid_pairs = 0

    # 2. Process Pairs
    for fname_l in left_images:
        fname_r = fname_l.replace("left_", "right_")

        if not os.path.exists(fname_r):
            continue

        # --- SWAP LOGIC IS HERE ---
        if SWAP_CAMERAS:
            # Load 'Right' file into 'Left' variable
            img_l = cv2.imread(fname_r)
            img_r = cv2.imread(fname_l)
        else:
            # Normal Load
            img_l = cv2.imread(fname_l)
            img_r = cv2.imread(fname_r)

        if frame_shape is None:
            frame_shape = img_l.shape[:2][::-1]

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret_l and ret_r:
            valid_pairs += 1
            print(f"[{valid_pairs}] Used: {os.path.basename(fname_l)}")

            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)

            objpoints.append(objp)
            imgpoints_l.append(corners2_l)
            imgpoints_r.append(corners2_r)

    if valid_pairs < 5:
        print("Error: Too few valid pairs found.")
        return

    # 3. Calibration
    print("\n--- RUNNING CALIBRATION ---")

    # ---------------------------------------------------------
    # A. INDIVIDUAL CALIBRATION
    # ---------------------------------------------------------
    print("Calibrating Left Camera...")
    # This line DEFINES mtx_l and dist_l needed later
    rms_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, frame_shape, None, None
    )
    print(f" >> Left RMS: {rms_l:.4f}")

    print("Calibrating Right Camera...")
    # This line DEFINES mtx_r and dist_r needed later
    rms_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, frame_shape, None, None
    )
    print(f" >> Right RMS: {rms_r:.4f}")

    # ---------------------------------------------------------
    # B. STEREO CALIBRATION
    # ---------------------------------------------------------
    print("Calibrating Stereo System...")

    # We pass mtx_l/mtx_r as input guesses because we used CALIB_USE_INTRINSIC_GUESS (or similar)
    (rms_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F) = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        frame_shape, flags=STEREO_FLAGS, criteria=CRITERIA
    )

    print(f"\n✅ Stereo RMS Error: {rms_stereo:.4f}")

    # --- SANITY CHECK FOR SWAP ---
    x_translation = T[0][0]
    print(f"Calculated Baseline (T[0]): {x_translation:.4f} meters")

    if x_translation > 0:
        print("\n⚠️ WARNING: T[0] is POSITIVE.")
        print("This usually means your cameras are SWAPPED.")
        print("Try setting SWAP_CAMERAS = True at the top of this script.")
    else:
        print("\nMeasurement looks correct (Right camera is to the right of Left).")

    # 4. Rectification
    print("Calculating Rectification Maps...")

    # alpha=0: Zoom in (Crop black borders)
    # alpha=1: Zoom out (Show all pixels + curved borders)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r,
        frame_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    data = {
        'cameraMatrix1': mtx_l, 'distCoeffs1': dist_l,
        'cameraMatrix2': mtx_r, 'distCoeffs2': dist_r,
        'R': R, 'T': T, 'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
        'frame_width': frame_shape[0], 'frame_height': frame_shape[1]
    }

    save_calibration_data(CALIB_FILE, data)
    print("Done.")


if __name__ == "__main__":
    main()