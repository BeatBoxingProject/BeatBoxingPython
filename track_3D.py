import cv2
import numpy as np
import time
import os
import socket
from threading import Thread

# ====================================================================
# --- 1. CONFIGURATION ---
# ====================================================================

url_1 = "http://192.168.137.102/stream"  # Left Camera
url_2 = "http://192.168.137.101/stream"  # Right Camera
CALIB_FILE = "stereo_calib.yml"

# Target Color
hsv_lower = np.array([145, 0, 135])
hsv_upper = np.array([174, 71, 255])

# UDP Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005


# ====================================================================
# --- 2. THE NEW THREADED CAMERA CLASS ---
# ====================================================================

class CameraStream:
    def __init__(self, src, name="Camera"):
        self.src = src
        self.name = name
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.reconnecting = False

        # Start the thread to read frames from the video stream
        # args=() means no arguments are passed to the update function
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # Thread dies when main program dies
        self.t.start()

    def update(self):
        # This loops runs in the background constantly
        while True:
            if self.stopped:
                self.stream.release()
                return

            # Try to read a frame
            ret, frame = self.stream.read()

            if ret:
                # If successful, update the 'latest' frame
                self.frame = frame
                self.ret = True
                self.reconnecting = False
            else:
                # If failed, handle reconnection
                self.ret = False
                if not self.reconnecting:
                    print(f"Stream error on {self.name}. Reconnecting...")
                    self.reconnecting = True
                    self.stream.release()
                    time.sleep(1)
                    self.stream = cv2.VideoCapture(self.src)

    def read(self):
        # Return the most recent frame read
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.t.join()


# ====================================================================
# --- 3. HELPER FUNCTIONS ---
# ====================================================================

def load_calibration_data(filename, target_width, target_height):
    """
    Loads calibration data and SCALES it to match the target resolution.
    """
    if not os.path.exists(filename):
        print(f"Error: Calibration file not found at {filename}")
        return None

    print(f"Loading calibration data from {filename}...")
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    # Load raw data
    calib_data = {}
    keys = ['cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2',
            'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q', 'frame_width', 'frame_height']
    for key in keys:
        node = fs.getNode(key)
        if node.isNone():
            continue
        calib_data[key] = node.mat() if key not in ['frame_width', 'frame_height'] else int(node.real())

    fs.release()

    # --- SCALING LOGIC ---
    orig_w = calib_data['frame_width']
    orig_h = calib_data['frame_height']

    if orig_w != target_width or orig_h != target_height:
        print(f"⚠️ SCALING CALIBRATION: {orig_w}x{orig_h} -> {target_width}x{target_height}")

        # Calculate scale factor (assuming aspect ratio is maintained)
        scale = target_width / orig_w

        # Scale the Camera Matrices (Intrinsic)
        calib_data['cameraMatrix1'] *= scale
        calib_data['cameraMatrix2'] *= scale

        # Restore the bottom-right element to 1.0 (Homogeneous coordinates requirement)
        calib_data['cameraMatrix1'][2, 2] = 1.0
        calib_data['cameraMatrix2'][2, 2] = 1.0

        # Scale the Projection Matrices (P1, P2) from stereoRectify
        # P matrices are 3x4. We scale the first 3 columns.
        calib_data['P1'] *= scale
        calib_data['P2'] *= scale
        calib_data['P1'][2, 2] = 1.0
        calib_data['P2'][2, 2] = 1.0

        # NOTE: distCoeffs, R, and T do NOT need scaling.

        # Update the width/height in the dict
        calib_data['frame_width'] = target_width
        calib_data['frame_height'] = target_height
    else:
        print("Resolution matches calibration. No scaling needed.")

    return calib_data


def find_target(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_target = None
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            found_target = (int(x), int(y), int(radius))
    return found_target, mask


# ====================================================================
# --- 4. MAIN APPLICATION ---
# ====================================================================

def main():
    # --- SETUP UDP ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP Target set to: {UDP_IP}:{UDP_PORT}")

    # --- 1. START THREADED STREAMS FIRST ---
    # We must start them now to see what resolution the ESP32 is actually sending
    print("Starting Camera Threads...")
    cam_l = CameraStream(url_1, "Left")
    cam_r = CameraStream(url_2, "Right")

    # Give threads a moment to connect and fill the buffer
    time.sleep(2.0)

    # --- 2. GET ACTUAL RESOLUTION ---
    print("Checking stream resolution...")
    ret, temp_frame = cam_l.read()

    # Retry loop in case cameras are slow to start
    tries = 0
    while not ret and tries < 10:
        print(f"Waiting for frame... ({tries}/10)")
        time.sleep(0.5)
        ret, temp_frame = cam_l.read()
        tries += 1

    if not ret:
        print("Error: Could not read frame from Left camera to determine resolution.")
        cam_l.stop()
        cam_r.stop()
        return

    stream_h, stream_w = temp_frame.shape[:2]
    print(f"Detected Stream Resolution: {stream_w}x{stream_h}")

    # --- 3. LOAD & SCALE CALIBRATION ---
    # Pass the detected width/height to the loader so it scales the math
    calib_data = load_calibration_data(CALIB_FILE, stream_w, stream_h)

    if calib_data is None:
        cam_l.stop()
        cam_r.stop()
        return

    # Extract matrices
    mtx_l, dist_l = calib_data['cameraMatrix1'], calib_data['distCoeffs1']
    mtx_r, dist_r = calib_data['cameraMatrix2'], calib_data['distCoeffs2']
    R1, R2, P1, P2 = calib_data['R1'], calib_data['R2'], calib_data['P1'], calib_data['P2']

    # Use the detected shape for mapping
    frame_shape = (stream_w, stream_h)

    # --- 4. RECTIFICATION MAPS ---
    print("Generating rectification maps...")
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, frame_shape, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, frame_shape, cv2.CV_32FC1)

    print("\nTracking started. Press 'q' to quit.")

    # --- 5. MAIN LOOP ---
    while True:
        try:
            # 1. Get latest frames from threads
            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()

            if not ret_l or not ret_r:
                time.sleep(0.01)
                continue

            # 2. Rectify
            rect_l = cv2.remap(frame_l, map_l_x, map_l_y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map_r_x, map_r_y, cv2.INTER_LINEAR)

            # 3. Find Targets
            target_l_data, mask_l = find_target(rect_l)
            target_r_data, mask_r = find_target(rect_r)

            # 4. Triangulate & Send
            if target_l_data and target_r_data:
                p1 = (target_l_data[0], target_l_data[1])
                p2 = (target_r_data[0], target_r_data[1])

                # Debug Circles
                cv2.circle(rect_l, p1, int(target_l_data[2]), (0, 255, 0), 2)
                cv2.circle(rect_r, p2, int(target_r_data[2]), (0, 255, 0), 2)

                # Math
                p1_np = np.array([[p1[0]], [p1[1]]], dtype=np.float64)
                p2_np = np.array([[p2[0]], [p2[1]]], dtype=np.float64)
                point_4d_hom = cv2.triangulatePoints(P1, P2, p1_np, p2_np)
                point_3d = point_4d_hom / point_4d_hom[3]

                x, y, z = point_3d[0][0], point_3d[1][0], point_3d[2][0]

                # Send UDP
                message = f"{x},{y},{z}"
                sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

                # Print (Optional, comment out for speed)
                # print(f"Sent: X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")

            # 5. Display
            cv2.imshow('Rectified Left', rect_l)
            cv2.imshow('Rectified Right', rect_r)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(0.1)

    # Cleanup
    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()