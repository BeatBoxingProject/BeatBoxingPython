import cv2
import glob
import os
import sys
import numpy as np

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import DATA_DIR

IMAGE_FOLDER = os.path.join(DATA_DIR, "calibration_images")
CHECKERBOARD = (9, 6)  # Ensure this matches your board inner corners


def process_image(img, name):
    """
    Helper to find and draw corners on a single image.
    Returns the processed image and a success flag.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH |
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        return img, True
    else:
        print(f"❌ Failed to detect corners in {name}")
        return img, False


def main():
    # 1. Find only the Left images first
    left_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "left_*.jpg")))

    if not left_images:
        print("No images found (looking for 'left_*.jpg').")
        return

    print(f"Found {len(left_images)} pairs. Press [SPACE] to next, [ESC] to quit.")

    for fname_l in left_images:
        # 2. Construct the matching Right filename
        fname_r = fname_l.replace("left_", "right_")

        if not os.path.exists(fname_r):
            print(f"Skipping {os.path.basename(fname_l)}: Right image missing.")
            continue

        # 3. Load Images
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)

        # 4. Process Both
        # We process the raw full-res image so drawing is accurate
        proc_l, success_l = process_image(img_l, "Left")
        proc_r, success_r = process_image(img_r, "Right")

        # 5. Create a Side-by-Side Display
        # Resize for display purposes so 2x UXGA images fit on screen
        display_h = 480  # Fixed height for preview
        scale = display_h / img_l.shape[0]
        display_w = int(img_l.shape[1] * scale)

        small_l = cv2.resize(proc_l, (display_w, display_h))
        small_r = cv2.resize(proc_r, (display_w, display_h))

        # Stack images horizontally
        combined = np.hstack((small_l, small_r))

        # Add labels
        cv2.putText(combined, f"L: {os.path.basename(fname_l)}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if success_l else (0, 0, 255), 2)
        cv2.putText(combined, f"R: {os.path.basename(fname_r)}", (display_w + 30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if success_r else (0, 0, 255), 2)

        # Show
        cv2.imshow("Stereo Debug (Left vs Right)", combined)

        # Wait for input
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()