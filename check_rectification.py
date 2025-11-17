import cv2
import numpy as np
import time
import os

# --- CONFIG ---
CALIB_FILE = "stereo_calib.yml"
url_1 = "http://192.168.137.102/stream"  # Left Camera
url_2 = "http://192.168.137.101/stream"  # Right Camera


# ------------

def load_calibration_data(filename):
    if not os.path.exists(filename):
        print(f"Error: Calibration file not found at {filename}")
        return None

    print(f"Loading calibration from {filename}...")
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"Error: Could not open {filename}")
        return None

    calib_data = {}
    # List of all matrices
    matrix_keys = [
        'cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2',
        'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q'
    ]
    # List of all single numbers
    value_keys = ['frame_width', 'frame_height']

    for key in matrix_keys:
        node = fs.getNode(key)
        if node.isNone():
            print(f"Error: Failed to read matrix key '{key}' from calibration file.")
            fs.release()
            return None
        calib_data[key] = node.mat()

    for key in value_keys:
        node = fs.getNode(key)
        if node.isNone():
            print(f"Error: Failed to read value key '{key}' from calibration file.")
            fs.release()
            return None
        calib_data[key] = node.real()

    # Cast numbers to int
    calib_data['frame_width'] = int(calib_data['frame_width'])
    calib_data['frame_height'] = int(calib_data['frame_height'])

    fs.release()
    print("Calibration data loaded successfully.")
    return calib_data


def connect_to_camera(url, camera_name):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Could not open {camera_name} stream at {url}")
        return None
    print(f"Connected to {camera_name}.")
    return cap


def main():
    calib_data = load_calibration_data(CALIB_FILE)
    if calib_data is None:
        print("Exiting due to calibration load failure.")
        return

    frame_shape = (calib_data['frame_width'], calib_data['frame_height'])

    # --- Check for Resolution Mismatch ---
    # Try to connect and grab one frame to check the size
    print("Checking for resolution mismatch...")
    temp_cap_l = connect_to_camera(url_1, "Left (Test)")
    if temp_cap_l:
        ret_l, frame_l = temp_cap_l.read()
        if ret_l:
            real_shape = frame_l.shape
            calib_shape = (calib_data['frame_height'], calib_data['frame_width'])  # (h, w)

            if real_shape[0] != calib_shape[0] or real_shape[1] != calib_shape[1]:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"CRITICAL WARNING: Resolution Mismatch!")
                print(f"Calibration was done at: {calib_shape[1]}x{calib_shape[0]} (WxH)")
                print(f"Camera is streaming at:  {real_shape[1]}x{real_shape[0]} (WxH)")
                print("This WILL cause a 'messed up' view. You MUST recalibrate")
                print("or change your camera's stream resolution to match.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                print("Stream resolution matches calibration file. Good.")
        temp_cap_l.release()
    # -------------------------------------

    print("Creating rectification maps...")
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        calib_data['cameraMatrix1'], calib_data['distCoeffs1'],
        calib_data['R1'], calib_data['P1'],
        frame_shape, cv2.CV_32FC1
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        calib_data['cameraMatrix2'], calib_data['distCoeffs2'],
        calib_data['R2'], calib_data['P2'],
        frame_shape, cv2.CV_32FC1
    )
    print("Maps created.")

    cap_l = connect_to_camera(url_1, "Left")
    cap_r = connect_to_camera(url_2, "Right")

    while True:
        try:
            if cap_l is None: cap_l = connect_to_camera(url_1, "Left")
            if cap_r is None: cap_r = connect_to_camera(url_2, "Right")

            if cap_l is None or cap_r is None:
                print("Waiting to connect...")
                time.sleep(2)
                continue

            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l: cap_l.release(); cap_l = None;
            if not ret_r: cap_r.release(); cap_r = None;
            if not ret_l or not ret_r:
                print("Frame grab error, resetting connection.")
                continue

            # --- THIS IS THE CORE OF THE TEST ---
            rect_l = cv2.remap(frame_l, map_l_x, map_l_y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map_r_x, map_r_y, cv2.INTER_LINEAR)
            # ------------------------------------

            # Draw horizontal lines on both frames to check alignment
            h, w, _ = rect_l.shape
            for i in range(1, 10):
                line_y = h * i // 10
                cv2.line(rect_l, (0, line_y), (w, line_y), (0, 255, 0), 1)
                cv2.line(rect_r, (0, line_y), (w, line_y), (0, 255, 0), 1)

            cv2.imshow('Rectified Left', rect_l)
            cv2.imshow('Rectified Right', rect_r)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error in main loop: {e}")
            if cap_l: cap_l.release()
            if cap_r: cap_r.release()
            cap_l, cap_r = None, None
            time.sleep(2)

    if cap_l: cap_l.release()
    if cap_r: cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()