import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
from core.config import LEFT_CAM_URL_TRACKING

# Global variables
current_frame = None
hsv_lower = np.array([0, 0, 0])
hsv_upper = np.array([179, 255, 255])

def nothing(x): pass

def pick_color(event, x, y, flags, param):
    global hsv_lower, hsv_upper, current_frame

    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        # 1. Get ROI
        y_min, y_max = max(0, y - 5), min(current_frame.shape[0], y + 5)
        x_min, x_max = max(0, x - 5), min(current_frame.shape[1], x + 5)
        roi = current_frame[y_min:y_max, x_min:x_max]
        if roi.size == 0: return

        # 2. Convert to HSV & Calc Min/Max
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        min_h, min_s, min_v = np.min(hsv_roi[:, :, 0]), np.min(hsv_roi[:, :, 1]), np.min(hsv_roi[:, :, 2])
        max_h, max_s, max_v = np.max(hsv_roi[:, :, 0]), np.max(hsv_roi[:, :, 1]), np.max(hsv_roi[:, :, 2])

        # 3. Set Tolerance (Int casting to prevent overflow)
        hsv_lower = np.array([int(max(0, min_h - 10)), int(max(0, min_s - 40)), int(max(0, min_v - 40))])
        hsv_upper = np.array([int(min(179, max_h + 10)), int(min(255, max_s + 40)), 255])

        print(f"Clicked! New Range: \nL: {hsv_lower} \nH: {hsv_upper}")

        # 4. Update Sliders
        for i, (l, u) in enumerate(zip(hsv_lower, hsv_upper)):
            cv2.setTrackbarPos(["L-H", "L-S", "L-V"][i], "Trackbars", l)
            cv2.setTrackbarPos(["U-H", "U-S", "U-V"][i], "Trackbars", u)

def main():
    print(f"Connecting to {LEFT_CAM_URL_TRACKING}...")
    cap = cv2.VideoCapture(LEFT_CAM_URL_TRACKING)
    
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 300)
    cv2.namedWindow("Picker")
    cv2.setMouseCallback("Picker", pick_color)

    # Sliders
    for label, default in [("L-H",0), ("L-S",0), ("L-V",0), ("U-H",179), ("U-S",255), ("U-V",255)]:
        cv2.createTrackbar(label, "Trackbars", default, 255 if 'H' not in label else 179, nothing)

    global current_frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconnecting...")
            time.sleep(1)
            cap = cv2.VideoCapture(LEFT_CAM_URL_TRACKING)
            continue

        height, width = frame.shape[:2]
        if width > 640:
            frame = cv2.resize(frame, (640, int(640 * (height / width))))
        
        current_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get Sliders
        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")

        mask = cv2.inRange(hsv, np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))
        
        cv2.imshow("Picker", frame)
        cv2.imshow("Mask Result", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'):
            print(f"hsv_lower = np.array([{l_h}, {l_s}, {l_v}])")
            print(f"hsv_upper = np.array([{u_h}, {u_s}, {u_v}])")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()