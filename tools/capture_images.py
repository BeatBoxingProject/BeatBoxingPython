import sys
import os
import cv2
import time
import numpy as np

# Allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import config to get URLs and Paths
from core.config import LEFT_CAM_URL_CALIB, RIGHT_CAM_URL_CALIB, DATA_DIR

# --- CONFIGURATION ---
# Where to save the images
SAVE_FOLDER = os.path.join(DATA_DIR, "calibration_images")
PREFIX_LEFT = "left_"
PREFIX_RIGHT = "right_"
EXT = ".jpg"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def main():
    # 1. Setup Folders
    ensure_dir(SAVE_FOLDER)
    print(f"Saving images to: {SAVE_FOLDER}")

    # 2. Connect to Cameras
    print(f"Connecting to Left: {LEFT_CAM_URL_CALIB}...")
    cap_l = cv2.VideoCapture(LEFT_CAM_URL_CALIB)

    print(f"Connecting to Right: {RIGHT_CAM_URL_CALIB}...")
    cap_r = cv2.VideoCapture(RIGHT_CAM_URL_CALIB)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("Error: Could not open one or both cameras.")
        return

    # 3. Determine next index (so we don't overwrite old photos)
    existing_files = [f for f in os.listdir(SAVE_FOLDER) if f.startswith(PREFIX_LEFT)]
    img_index = len(existing_files) + 1

    print("\n--- CONTROLS ---")
    print(" [SPACE]  : Capture synchronized image pair")
    print(" [Q]      : Quit")
    print("----------------")

    while True:
        # Read frames
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            print("Warning: Dropped frame from one camera.")
            time.sleep(0.1)
            continue

        # Resize for display (so it fits on screen)
        # We save the FULL resolution 'frame_l', but show a smaller 'display_l'
        display_h, display_w = int(frame_l.shape[0] * 0.5), int(frame_l.shape[1] * 0.5)
        display_l = cv2.resize(frame_l, (display_w, display_h))
        display_r = cv2.resize(frame_r, (display_w, display_h))

        # Show status
        cv2.imshow("Left Camera (Preview)", display_l)
        cv2.imshow("Right Camera (Preview)", display_r)

        key = cv2.waitKey(1) & 0xFF

        # --- CAPTURE LOGIC ---
        if key == ord(' '):
            # Filenames
            filename_l = f"{PREFIX_LEFT}{img_index:02d}{EXT}"
            filename_r = f"{PREFIX_RIGHT}{img_index:02d}{EXT}"
            path_l = os.path.join(SAVE_FOLDER, filename_l)
            path_r = os.path.join(SAVE_FOLDER, filename_r)

            # Save the raw full-resolution frames
            cv2.imwrite(path_l, frame_l)
            cv2.imwrite(path_r, frame_r)

            print(f"Saved Set #{img_index}: {filename_l} & {filename_r}")

            # Visual Feedback (Flash White)
            cv2.imshow("Left Camera (Preview)", np.ones_like(display_l) * 255)
            cv2.waitKey(50)

            img_index += 1

        elif key == ord('q'):
            break

    # Cleanup
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print("Capture session finished.")


if __name__ == "__main__":
    main()