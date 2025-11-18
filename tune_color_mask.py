import cv2
import numpy as np
import time

# ==========================================
# CONFIG
# ==========================================
URL = "http://192.168.137.102/stream"
# ==========================================

# Global variables to store the latest frame and current HSV range
current_frame = None
hsv_lower = np.array([0, 0, 0])
hsv_upper = np.array([179, 255, 255])


def nothing(x):
    pass


def pick_color(event, x, y, flags, param):
    global hsv_lower, hsv_upper, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame is None: return

        # 1. Get ROI (Region of Interest)
        y_min = max(0, y - 5)
        y_max = min(current_frame.shape[0], y + 5)
        x_min = max(0, x - 5)
        x_max = min(current_frame.shape[1], x + 5)

        roi = current_frame[y_min:y_max, x_min:x_max]
        if roi.size == 0: return

        # 2. Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3. Calculate Min/Max AND CAST TO INT (The Fix!)
        # We use int() to prevent "uint8" overflow/underflow
        min_h = int(np.min(hsv_roi[:, :, 0]))
        min_s = int(np.min(hsv_roi[:, :, 1]))
        min_v = int(np.min(hsv_roi[:, :, 2]))

        max_h = int(np.max(hsv_roi[:, :, 0]))
        max_s = int(np.max(hsv_roi[:, :, 1]))
        max_v = int(np.max(hsv_roi[:, :, 2]))

        # 4. Tolerance padding
        # Now that they are ints, negative numbers stay negative,
        # so max(0, -20) correctly becomes 0.
        hsv_lower = np.array([max(0, min_h - 10), max(0, min_s - 40), max(0, min_v - 40)])
        hsv_upper = np.array([min(179, max_h + 10), min(255, max_s + 40), 255])

        print(f"Clicked! Auto-setting range to: \nLower: {hsv_lower} \nUpper: {hsv_upper}")

        # 5. Update Sliders
        cv2.setTrackbarPos("L - H", "Trackbars", hsv_lower[0])
        cv2.setTrackbarPos("L - S", "Trackbars", hsv_lower[1])
        cv2.setTrackbarPos("L - V", "Trackbars", hsv_lower[2])
        cv2.setTrackbarPos("U - H", "Trackbars", hsv_upper[0])
        cv2.setTrackbarPos("U - S", "Trackbars", hsv_upper[1])
        cv2.setTrackbarPos("U - V", "Trackbars", hsv_upper[2])


# Create Windows
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 300)
cv2.namedWindow("Picker")

# Set the Mouse Callback
cv2.setMouseCallback("Picker", pick_color)

# Create Sliders
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

print(f"Connecting to {URL}...")
cap = cv2.VideoCapture(URL)

# Robust Loop Variables
fail_count = 0
FAIL_THRESHOLD = 10

while True:
    try:
        ret, frame = cap.read()

        # --- Reconnect Logic ---
        if not ret:
            fail_count += 1
            if fail_count >= FAIL_THRESHOLD:
                print("Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(URL)
                fail_count = 0
            time.sleep(0.1)
            continue
        fail_count = 0
        # -----------------------

        if frame is None: continue

        # Resize and flip for easier usage
        height, width = frame.shape[:2]
        if width > 640:
            frame = cv2.resize(frame, (640, int(640 * (height / width))))

        # Update global frame for the mouse click function
        current_frame = frame.copy()

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current slider positions (User can tweak these after clicking)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])

        # Create Mask
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # Show results
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # We show the Original frame in the "Picker" window so you can click it
        cv2.imshow("Picker", frame)
        # We show the Mask separately
        cv2.imshow("Mask Result", mask_3ch)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            print(f"\n=== COPY THESE VALUES ===")
            print(f"hsv_lower = np.array([{l_h}, {l_s}, {l_v}])")
            print(f"hsv_upper = np.array([{u_h}, {u_s}, {u_v}])")
            print("=========================\n")

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)

cap.release()
cv2.destroyAllWindows()