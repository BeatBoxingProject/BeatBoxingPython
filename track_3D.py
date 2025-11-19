import cv2
import numpy as np
import time
import os
import socket
from threading import Thread
import math

# ====================================================================
# --- 1. CONFIGURATION ---
# ====================================================================

url_1 = "http://192.168.137.101/stream?framesize=8"  # Left Camera
url_2 = "http://192.168.137.102/stream?framesize=8"  # Right Camera
CALIB_FILE = "stereo_calib.yml"

# Target Color
hsv_lower = np.array([26, 41, 168])
hsv_upper = np.array([79, 255, 255])

# UDP Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# --- NEW: TILT CORRECTION ---
# Estimate how many degrees the cameras are angled DOWN from the horizon.
# 0 = Looking straight forward (Horizontal)
# 90 = Looking straight down (Vertical)
# 45 = Typical ceiling mount angle
CAMERA_TILT_ANGLE = 50  # <--- ADJUST THIS VALUE


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
                    print(f"Stream error on {self.name}. Reconnecting...")
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
# --- 3. HELPER FUNCTIONS ---
# ====================================================================

def apply_tilt_correction(x, y, z, angle_degrees):
    """
    Rotates the 3D point around the X-axis to compensate for camera tilt.
    Converts 'Camera Space' to 'World Space'.
    """
    # Convert to radians
    theta = math.radians(angle_degrees)

    # Standard 3D Rotation Matrix around X-axis (counter-clockwise)
    # We want to rotate the 'world' UP, so we use negative theta effectively
    # OpenCV Coords: Y is DOWN, Z is FORWARD

    # Formula for rotating coordinate system tilted down by theta:
    # y_new = y * cos(theta) - z * sin(theta)
    # z_new = y * sin(theta) + z * cos(theta)

    # Note: Because OpenCV Y is "Down", we might need to invert logic depending on
    # exactly how you want "World Y" (Height) to behave.
    # Assuming we want Y to be UP (like Unity) and Z to be Forward (Parallel to ground)

    # 1. First, let's stick to OpenCV format (Y down) but rotated
    y_new = y * math.cos(theta) - z * math.sin(theta)
    z_new = y * math.sin(theta) + z * math.cos(theta)

    return x, y_new, z_new


def load_calibration_data(filename, target_width, target_height):
    if not os.path.exists(filename):
        print(f"Error: Calibration file not found at {filename}")
        return None
    print(f"Loading calibration data from {filename}...")
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    calib_data = {}
    keys = ['cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2',
            'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q', 'frame_width', 'frame_height']
    for key in keys:
        node = fs.getNode(key)
        if node.isNone(): continue
        calib_data[key] = node.mat() if key not in ['frame_width', 'frame_height'] else int(node.real())
    fs.release()

    orig_w = calib_data['frame_width']
    orig_h = calib_data['frame_height']
    if orig_w != target_width or orig_h != target_height:
        print(f"⚠️ SCALING CALIBRATION: {orig_w}x{orig_h} -> {target_width}x{target_height}")
        scale = target_width / orig_w
        calib_data['cameraMatrix1'] *= scale
        calib_data['cameraMatrix2'] *= scale
        calib_data['cameraMatrix1'][2, 2] = 1.0
        calib_data['cameraMatrix2'][2, 2] = 1.0
        calib_data['P1'] *= scale
        calib_data['P2'] *= scale
        calib_data['P1'][2, 2] = 1.0
        calib_data['P2'][2, 2] = 1.0
        calib_data['frame_width'] = target_width
        calib_data['frame_height'] = target_height
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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP Target set to: {UDP_IP}:{UDP_PORT}")

    print("Starting Camera Threads...")
    cam_l = CameraStream(url_1, "Left")
    cam_r = CameraStream(url_2, "Right")
    time.sleep(2.0)

    print("Checking stream resolution...")
    ret, temp_frame = cam_l.read()
    tries = 0
    while not ret and tries < 10:
        time.sleep(0.5)
        ret, temp_frame = cam_l.read()
        tries += 1

    if not ret:
        print("Error: Could not read frame.")
        return

    stream_h, stream_w = temp_frame.shape[:2]
    print(f"Detected Stream Resolution: {stream_w}x{stream_h}")

    calib_data = load_calibration_data(CALIB_FILE, stream_w, stream_h)
    if calib_data is None: return

    mtx_l, dist_l = calib_data['cameraMatrix1'], calib_data['distCoeffs1']
    mtx_r, dist_r = calib_data['cameraMatrix2'], calib_data['distCoeffs2']
    R1, R2, P1, P2 = calib_data['R1'], calib_data['R2'], calib_data['P1'], calib_data['P2']
    frame_shape = (stream_w, stream_h)

    print("Generating rectification maps...")
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, frame_shape, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, frame_shape, cv2.CV_32FC1)

    print("\nTracking started. Press 'q' to quit.")

    while True:
        try:
            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()

            if not ret_l or not ret_r:
                time.sleep(0.01)
                continue

            rect_l = cv2.remap(frame_l, map_l_x, map_l_y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map_r_x, map_r_y, cv2.INTER_LINEAR)

            target_l_data, mask_l = find_target(rect_l)
            target_r_data, mask_r = find_target(rect_r)

            if target_l_data and target_r_data:
                p1 = (target_l_data[0], target_l_data[1])
                p2 = (target_r_data[0], target_r_data[1])

                cv2.circle(rect_l, p1, int(target_l_data[2]), (0, 255, 0), 2)
                cv2.circle(rect_r, p2, int(target_r_data[2]), (0, 255, 0), 2)

                p1_np = np.array([[p1[0]], [p1[1]]], dtype=np.float64)
                p2_np = np.array([[p2[0]], [p2[1]]], dtype=np.float64)
                point_4d_hom = cv2.triangulatePoints(P1, P2, p1_np, p2_np)
                point_3d = point_4d_hom / point_4d_hom[3]

                raw_x, raw_y, raw_z = point_3d[0][0], point_3d[1][0], point_3d[2][0]

                # --- APPLY TILT CORRECTION HERE ---
                final_x, final_y, final_z = apply_tilt_correction(raw_x, raw_y, raw_z, CAMERA_TILT_ANGLE)

                message = f"{final_x},{final_y},{final_z}"
                sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

                # Debug Print: Compare Raw vs Corrected
                print(f"Raw X: {raw_x:.2f} | Corrected X: {final_x:.2f}")
                print(f"Raw Y: {raw_y:.2f} | Corrected Y: {final_y:.2f}")
                print(f"Raw Z: {raw_z:.2f} | Corrected Z: {final_z:.2f}")
                print("-----------------------------------------------")

            cv2.imshow('Rectified Left', rect_l)
            cv2.imshow('Rectified Right', rect_r)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(0.1)

    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()