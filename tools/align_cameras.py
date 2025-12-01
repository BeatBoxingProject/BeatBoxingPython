import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
from core.camera import CameraStream
from core.config import LEFT_CAM_URL_TRACKING, RIGHT_CAM_URL_TRACKING

def draw_crosshair(img):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    color = (0, 255, 0)
    
    cv2.circle(img, (cx, cy), 20, color, 2)
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv2.line(img, (cx - 40, cy), (cx + 40, cy), color, 2)
    cv2.line(img, (cx, cy - 40), (cx, cy + 40), color, 2)
    return img

def main():
    print("Connecting to cameras...")
    cam_l = CameraStream(LEFT_CAM_URL_TRACKING, "Left")
    cam_r = CameraStream(RIGHT_CAM_URL_TRACKING, "Right")
    time.sleep(2.0)

    print("\nAlignment Tool Started.")
    print("Adjust cameras so Green Crosshairs overlap physically.")

    while True:
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()

        if not ret_l or not ret_r:
            time.sleep(0.1)
            continue

        display_l = draw_crosshair(frame_l.copy())
        display_r = draw_crosshair(frame_r.copy())

        cv2.putText(display_l, "LEFT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_r, "RIGHT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if display_l.shape[0] != display_r.shape[0]:
            display_r = cv2.resize(display_r, (display_l.shape[1], display_l.shape[0]))

        combined = np.hstack((display_l, display_r))
        cv2.imshow("Camera Alignment Tool", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()