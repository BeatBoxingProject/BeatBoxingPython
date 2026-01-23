import cv2
import glob
import os
import sys

# Setup paths (Adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import DATA_DIR

IMAGE_FOLDER = os.path.join(DATA_DIR, "calibration_images")
CHECKERBOARD = (9, 6)  # Ensure this matches your board


def main():
    images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))

    if not images:
        print("No images found.")
        return

    print(f"Checking {len(images)} images... Press [SPACE] to verify, [ESC] to quit.")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH |
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            # Draw them
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

            # Zoom in to the center to see pixel-level accuracy
            h, w = img.shape[:2]
            center_x, center_y = w // 2, h // 2
            zoom_size = 200

            # Ensure crop is within bounds
            y1 = max(0, center_y - zoom_size)
            y2 = min(h, center_y + zoom_size)
            x1 = max(0, center_x - zoom_size)
            x2 = min(w, center_x + zoom_size)

            # Show full image
            cv2.imshow("Full View", img)

            print(f"Viewing: {os.path.basename(fname)}")
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
        else:
            print(f"❌ Failed to detect corners in: {os.path.basename(fname)}")
            cv2.imshow("Full View", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()