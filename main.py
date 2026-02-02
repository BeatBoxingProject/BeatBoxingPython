import cv2
import time
import socket
import numpy as np
from core.config import *
from core.camera import CameraStream
from core.vision import CoordinateSmoother, find_target, apply_tilt_correction, load_calibration_data
from core.visualizer import draw_visualizer

# --- CONFIG CHECK ---
# Ensure distinct colors are defined for each hand.
# If these are missing from your config, define them here or add them to config.py
try:
    _ = HSV_LEFT_LOWER
except NameError:
    print("⚠️  Warning: Left/Right HSV not found in config. Using defaults.")
    # Example: Pink for Left
    HSV_LEFT_LOWER = np.array([140, 50, 50])
    HSV_LEFT_UPPER = np.array([170, 255, 255])
    # Example: Blue/Green for Right
    HSV_RIGHT_LOWER = np.array([35, 50, 50])
    HSV_RIGHT_UPPER = np.array([85, 255, 255])


def main():
    # 1. Setup Network
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP Target set to: {UDP_IP}:{UDP_PORT}")

    # 2. Start Cameras
    print("Starting Camera Threads...")
    cam_l = CameraStream(LEFT_CAM_URL_TRACKING, "Left")
    cam_r = CameraStream(RIGHT_CAM_URL_TRACKING, "Right")
    time.sleep(2.0)  # Allow warmup

    # 3. Check Resolution & Load Calibration
    print("Checking stream resolution...")
    ret, temp_frame = cam_l.read()
    while not ret:
        time.sleep(0.5)
        ret, temp_frame = cam_l.read()

    stream_h, stream_w = temp_frame.shape[:2]
    print(f"Detected Stream Resolution: {stream_w}x{stream_h}")

    calib_data = load_calibration_data(CALIB_FILE, stream_w, stream_h)
    if calib_data is None:
        print("Calibration data missing. Please run tools/calibrate.py first.")
        return

    # Unpack Calibration
    mtx_l, dist_l = calib_data['cameraMatrix1'], calib_data['distCoeffs1']
    mtx_r, dist_r = calib_data['cameraMatrix2'], calib_data['distCoeffs2']
    R1, R2, P1, P2 = calib_data['R1'], calib_data['R2'], calib_data['P1'], calib_data['P2']
    frame_shape = (stream_w, stream_h)

    # 4. Generate Rectification Maps
    print("Generating rectification maps...")
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, frame_shape, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, frame_shape, cv2.CV_32FC1)

    print("\nTracking started. Press 'q' to quit.")

    # 5. Initialize Logic
    # We need separate smoothers for Left and Right hands
    smoother_L = CoordinateSmoother(buffer_size=SMOOTHING_BUFFER_SIZE)
    smoother_R = CoordinateSmoother(buffer_size=SMOOTHING_BUFFER_SIZE)

    # State variables for persistence (Default to 0 if hand is lost)
    pos_L = (0.0, 0.0, 0.0)
    pos_R = (0.0, 0.0, 0.0)

    # Helper function to process a single hand's logic
    def process_hand(target_data_l, target_data_r, smoother_obj):
        if target_data_l and target_data_r:
            p1 = (target_data_l[0], target_data_l[1])
            p2 = (target_data_r[0], target_data_r[1])

            # Draw circles on frames (green for valid pair)
            cv2.circle(rect_l, p1, int(target_data_l[2]), (0, 255, 0), 2)
            cv2.circle(rect_r, p2, int(target_data_r[2]), (0, 255, 0), 2)

            # Triangulate
            p1_np = np.array([[p1[0]], [p1[1]]], dtype=np.float64)
            p2_np = np.array([[p2[0]], [p2[1]]], dtype=np.float64)
            point_4d_hom = cv2.triangulatePoints(P1, P2, p1_np, p2_np)
            point_3d = point_4d_hom / point_4d_hom[3]

            raw_x, raw_y, raw_z = point_3d[0][0], point_3d[1][0], point_3d[2][0]

            # Transform: Tilt -> Swap Axes (Unity)
            cx, cy, cz = apply_tilt_correction(raw_x, raw_y, raw_z, CAMERA_TILT_ANGLE)
            # Mapping: X->X, Z->Y (Height), Y->Z (Depth)
            swapped_x, swapped_y, swapped_z = cx, cz, cy

            return smoother_obj.update(swapped_x, swapped_y, swapped_z)
        return None

    while True:
        try:
            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()

            if not ret_l or not ret_r:
                time.sleep(0.01)
                continue

            # Rectify Images
            rect_l = cv2.remap(frame_l, map_l_x, map_l_y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map_r_x, map_r_y, cv2.INTER_LINEAR)

            # --- TRACK LEFT HAND ---
            l_data_L, mask_l_L = find_target(rect_l, HSV_LEFT_LOWER, HSV_LEFT_UPPER)
            r_data_L, mask_r_L = find_target(rect_r, HSV_LEFT_LOWER, HSV_LEFT_UPPER)

            new_pos_L = process_hand(l_data_L, r_data_L, smoother_L)
            if new_pos_L:
                pos_L = new_pos_L  # Update persistence only if found

            # --- TRACK RIGHT HAND ---
            l_data_R, mask_l_R = find_target(rect_l, HSV_RIGHT_LOWER, HSV_RIGHT_UPPER)
            r_data_R, mask_r_R = find_target(rect_r, HSV_RIGHT_LOWER, HSV_RIGHT_UPPER)

            new_pos_R = process_hand(l_data_R, r_data_R, smoother_R)
            if new_pos_R:
                pos_R = new_pos_R

            # --- SEND DATA ---
            # Format: "Lx,Ly,Lz|Rx,Ry,Rz"
            # We use 2 decimal places to keep packet size small
            message = f"{pos_L[0]:.2f},{pos_L[1]:.2f},{pos_L[2]:.2f}|{pos_R[0]:.2f},{pos_R[1]:.2f},{pos_R[2]:.2f}"
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

            # --- VISUALIZE ---
            # Note: Visualizer currently only draws one point.
            # Update 'draw_visualizer' to accept two sets of coordinates if needed.
            vis_frame = draw_visualizer(pos_L[0], pos_L[1], pos_L[2])

            cv2.imshow('3D Data', vis_frame)
            cv2.imshow('Left Eye', rect_l)
            cv2.imshow('Right Eye', rect_r)

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