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
CHECKERBOARD = (9, 6)
SQUARE_SIZE_METERS = 0.025
IMAGE_FOLDER = os.path.join(DATA_DIR, "calibration_images")

# --- CAMERA ORDER SETTING ---
SWAP_CAMERAS = False

INDIVIDUAL_FLAGS = (
    cv2.CALIB_FIX_PRINCIPAL_POINT |
    cv2.CALIB_FIX_ASPECT_RATIO |
    cv2.CALIB_ZERO_TANGENT_DIST    # ESP32 sensors are flat, so this is usually 0
)

STEREO_FLAGS = (
    cv2.CALIB_FIX_INTRINSIC |      # TRUST the individual calibration! Don't warp it further.
    cv2.CALIB_FIX_PRINCIPAL_POINT |
    cv2.CALIB_FIX_ASPECT_RATIO |
    cv2.CALIB_ZERO_TANGENT_DIST |
    cv2.CALIB_SAME_FOCAL_LENGTH
)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


# ====================================================================
# --- HELPERS ---
# ====================================================================

def calculate_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist, filenames):
    """
    Calculates the error for each specific image to find 'Bad Apples'.
    """
    total_error = 0
    error_list = []

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

        # Store error with filename for sorting
        name = os.path.basename(filenames[i])
        error_list.append((name, error))

    return error_list


def print_bad_apples(error_list, label):
    """
    Prints the list of images sorted by worst error.
    """
    # Sort by error (descending)
    sorted_errors = sorted(error_list, key=lambda x: x[1], reverse=True)

    print(f"\n--- {label} PER-IMAGE ERRORS (Worst to Best) ---")
    print(f"{'Filename':<30} | {'Error':<10} |Status")
    print("-" * 55)

    bad_count = 0
    for name, err in sorted_errors:
        status = "✅ OK"
        if err > 1.0:
            status = "❌ BAD"
            bad_count += 1
        elif err > 0.5:
            status = "⚠️ WEAK"

        print(f"{name:<30} | {err:.4f}     | {status}")

    if bad_count > 0:
        print(f"\n💡 TIP: Delete the {bad_count} images marked '❌ BAD' and run this again.")
    print("-" * 55)


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
    print("Successfully saved data.")


# ====================================================================
# --- MAIN ---
# ====================================================================

def main():
    print(f"--- STEREO CALIBRATION WITH ERROR CHECKING ---")
    print(f"Searching in: {IMAGE_FOLDER}")

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

    # Keep track of which files we actually used
    used_filenames_l = []
    used_filenames_r = []

    frame_shape = None
    valid_pairs = 0

    # 1. Process Pairs
    for fname_l in left_images:
        fname_r = fname_l.replace("left_", "right_")

        if not os.path.exists(fname_r):
            continue

        if SWAP_CAMERAS:
            img_l = cv2.imread(fname_r)
            img_r = cv2.imread(fname_l)
            name_l = fname_r
            name_r = fname_l
        else:
            img_l = cv2.imread(fname_l)
            img_r = cv2.imread(fname_r)
            name_l = fname_l
            name_r = fname_r

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

            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)

            objpoints.append(objp)
            imgpoints_l.append(corners2_l)
            imgpoints_r.append(corners2_r)

            used_filenames_l.append(name_l)
            used_filenames_r.append(name_r)

    if valid_pairs < 5:
        print("Error: Too few valid pairs found.")
        return

    print(f"Used {valid_pairs} valid image pairs.")

    # 2. Individual Calibration & Error Checking
    print("\n--- ANALYZING INDIVIDUAL CAMERAS ---")

    print("Calibrating Left Camera...")
    rms_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, frame_shape, None, None, flags=INDIVIDUAL_FLAGS
    )
    # CHECK PER-IMAGE ERRORS LEFT
    errors_l = calculate_reprojection_errors(objpoints, imgpoints_l, rvecs_l, tvecs_l, mtx_l, dist_l, used_filenames_l)
    print_bad_apples(errors_l, "LEFT CAMERA")

    print("\nCalibrating Right Camera...")
    rms_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, frame_shape, None, None, flags=INDIVIDUAL_FLAGS
    )
    # CHECK PER-IMAGE ERRORS RIGHT
    errors_r = calculate_reprojection_errors(objpoints, imgpoints_r, rvecs_r, tvecs_r, mtx_r, dist_r, used_filenames_r)
    print_bad_apples(errors_r, "RIGHT CAMERA")

    # 3. Stereo Calibration
    print("\n--- STEREO CALIBRATION ---")
    (rms_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F) = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        frame_shape, flags=STEREO_FLAGS, criteria=CRITERIA
    )

    print(f"\n✅ Final Stereo RMS Error: {rms_stereo:.4f}")

    x_translation = T[0][0]
    print(f"Calculated Baseline: {x_translation:.4f} meters")

    # 4. Rectification
    print("Rectifying...")
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
    print("Done.")


if __name__ == "__main__":
    main()