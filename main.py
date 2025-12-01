import cv2
import time
import socket
import numpy as np
from core.config import *
from core.camera import CameraStream
from core.vision import CoordinateSmoother, find_target, apply_tilt_correction, load_calibration_data
from core.visualizer import draw_visualizer

def main():
    # 1. Setup Network
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP Target set to: {UDP_IP}:{UDP_PORT}")

    # 2. Start Cameras
    print("Starting Camera Threads...")
    cam_l = CameraStream(LEFT_CAM_URL_TRACKING, "Left")
    cam_r = CameraStream(RIGHT_CAM_URL_TRACKING, "Right")
    time.sleep(2.0) # Allow warmup

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
    smoother = CoordinateSmoother(buffer_size=SMOOTHING_BUFFER_SIZE)
    
    # State variables for persistence
    final_x, final_y, final_z = 0, 0, 0

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

            # Find Targets
            target_l_data, mask_l = find_target(rect_l, HSV_LOWER, HSV_UPPER)
            target_r_data, mask_r = find_target(rect_r, HSV_LOWER, HSV_UPPER)

            if target_l_data and target_r_data:
                p1 = (target_l_data[0], target_l_data[1])
                p2 = (target_r_data[0], target_r_data[1])

                # Draw debug circles
                cv2.circle(rect_l, p1, int(target_l_data[2]), (0, 255, 0), 2)
                cv2.circle(rect_r, p2, int(target_r_data[2]), (0, 255, 0), 2)

                # Triangulate
                p1_np = np.array([[p1[0]], [p1[1]]], dtype=np.float64)
                p2_np = np.array([[p2[0]], [p2[1]]], dtype=np.float64)
                point_4d_hom = cv2.triangulatePoints(P1, P2, p1_np, p2_np)
                point_3d = point_4d_hom / point_4d_hom[3]

                raw_x, raw_y, raw_z = point_3d[0][0], point_3d[1][0], point_3d[2][0]

                # Transform Coordinates
                # 1. Tilt Correction
                cx, cy, cz = apply_tilt_correction(raw_x, raw_y, raw_z, CAMERA_TILT_ANGLE)

                # 2. Swap Axes for Unity (Unity Y is Up, Unity Z is Forward)
                # We map: Python X -> Unity X, Python Z -> Unity Y, Python Y -> Unity Z
                swapped_x, swapped_y, swapped_z = cx, cz, cy

                # 3. Smoothing
                final_x, final_y, final_z = smoother.update(swapped_x, swapped_y, swapped_z)

                # Send Data
                message = f"{final_x},{final_y},{final_z}"
                sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

            # Visualization
            vis_frame = draw_visualizer(final_x, final_y, final_z)
            cv2.imshow('3D Data Visualizer', vis_frame)
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