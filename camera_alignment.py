import cv2
import numpy as np
import time
from threading import Thread

# ====================================================================
# --- CONFIGURATION ---
# ====================================================================

# Using Framesize 8 (VGA 640x480) for smooth alignment
# If you want higher res, change to 10 (XGA) or 13 (UXGA)
FRAME_SIZE = 9

url_1 = f"http://192.168.137.101/stream?framesize={FRAME_SIZE}"  # Left Camera
url_2 = f"http://192.168.137.102/stream?framesize={FRAME_SIZE}"  # Right Camera


# ====================================================================
# --- THREADED CAMERA CLASS (Prevents lag) ---
# ====================================================================

class CameraStream:
    def __init__(self, src, name="Camera"):
        self.src = src
        self.name = name
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.reconnecting = False

        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            ret, frame = self.stream.read()
            if ret:
                self.frame = frame
                self.ret = True
                self.reconnecting = False
            else:
                self.ret = False
                if not self.reconnecting:
                    # print(f"Stream error on {self.name}. Reconnecting...") # Optional log
                    self.reconnecting = True
                    self.stream.release()
                    time.sleep(1)
                    self.stream = cv2.VideoCapture(self.src)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.t.join()


# ====================================================================
# --- MAIN ALIGNMENT LOOP ---
# ====================================================================

def draw_crosshair(img):
    """Draws a target crosshair in the center of the image."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Color: Bright Green
    color = (0, 255, 0)
    thickness = 2

    # Draw Circle
    cv2.circle(img, (cx, cy), 20, color, thickness)
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)  # Red center dot

    # Draw Cross lines (extending slightly beyond circle)
    # Horizontal
    cv2.line(img, (cx - 40, cy), (cx + 40, cy), color, thickness)
    # Vertical
    cv2.line(img, (cx, cy - 40), (cx, cy + 40), color, thickness)

    return img


def main():
    print(f"Connecting to Left: {url_1}")
    print(f"Connecting to Right: {url_2}")
    print("Waiting for streams...")

    cam_l = CameraStream(url_1, "Left")
    cam_r = CameraStream(url_2, "Right")

    # Allow cameras to warm up
    time.sleep(2.0)

    print("\nAlignment Tool Started.")
    print("Adjust your cameras so the Green Crosshairs point at the EXACT same spot.")
    print("Press 'q' to quit.")

    while True:
        # 1. Get frames
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()

        if not ret_l or not ret_r:
            # If one camera is down, show a blank screen or waiting message
            time.sleep(0.1)
            continue

        # 2. Draw Crosshairs on both
        # We use .copy() so we don't draw on the internal buffer frame
        display_l = draw_crosshair(frame_l.copy())
        display_r = draw_crosshair(frame_r.copy())

        # 3. Add Labels
        cv2.putText(display_l, "LEFT CAMERA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_r, "RIGHT CAMERA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 4. Stack side-by-side
        # Ensure they are same height before stacking
        if display_l.shape[0] != display_r.shape[0]:
            display_r = cv2.resize(display_r, (display_l.shape[1], display_l.shape[0]))

        combined = np.hstack((display_l, display_r))

        # 5. Show
        cv2.imshow("Camera Alignment Tool", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()