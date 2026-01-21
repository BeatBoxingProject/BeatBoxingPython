import cv2
import time
import socket
from core.config import *
from core.camera import CameraStream
from core.vision import CoordinateSmoother, find_target, apply_tilt_correction, load_calibration_data
from core.visualizer import draw_visualizer


def process_hand(rect_l, rect_r, hsv_lower, hsv_upper, P1, P2, smoother, tilt_angle):
    """
    Helper function to find ONE hand, triangulate it, and smooth it.
    Returns (x, y, z) or None if not found.
    """
    # 1. Find targets in both eyes
    target_l, _ = find_target(rect_l, hsv_lower, hsv_upper)
    target_r, _ = find_target(rect_r, hsv_lower, hsv_upper)

    if target_l and target_r:
        # 2. Draw debug circles (Visual feedback)
        cv2.circle(rect_l, (target_l[0], target_l[1]), int(target_l[2]), (0, 255, 255), 2)
        cv2.circle(rect_r, (target_r[0], target_r[1]), int(target_r[2]), (0, 255, 255), 2)

        # 3. Triangulate
        p1_np = np.array([[target_l[0]], [target_l[1]]], dtype=np.float64)
        p2_np = np.array([[target_r[0]], [target_r[1]]], dtype=np.float64)
        point_4d_hom = cv2.triangulatePoints(P1, P2, p1_np, p2_np)
        point_3d = point_4d_hom / point_4d_hom[3]

        raw_x, raw_y, raw_z = point_3d[0][0], point_3d[1][0], point_3d[2][0]

        # 4. Transform (Tilt + Swap Axes)
        cx, cy, cz = apply_tilt_correction(raw_x, raw_y, raw_z, tilt_angle)

        # Mapping: Py X->Unity X, Py Z->Unity Y, Py Y->Unity Z
        swapped_x, swapped_y, swapped_z = cx, cz, cy

        # 5. Smooth
        return smoother.update(swapped_x, swapped_y, swapped_z)

    return None


def main():
    # 1. Setup Network
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP Target set to: {UDP_IP}:{UDP_PORT}")

    # 2. Start Cameras
    print("Starting Camera Threads...")
    cam_l = CameraStream(LEFT_CAM_URL_TRACKING, "Left")
    cam_r = CameraStream(RIGHT_CAM_URL_TRACKING, "Right")
    time.sleep(2.0)

    # 3. Load Calibration
    print("Checking resolution...")
    ret, temp_frame = cam_l.read()
    while not ret:
        time.sleep(0.5)
        ret, temp_frame = cam_l.read()

    h, w = temp_frame.shape[:2]
    calib_data = load_calibration_data(CALIB_FILE, w, h)
    if calib_data is None: return

    mtx_l, dist_l = calib_data['cameraMatrix1'], calib_data['distCoeffs1']
    mtx_r, dist_r = calib_data['cameraMatrix2'], calib_data['distCoeffs2']
    R1, R2, P1, P2 = calib_data['R1'], calib_data['R2'], calib_data['P1'], calib_data['P2']

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_32FC1)

    # 4. Initialize Smoothers (ONE PER HAND)
    smoother_left = CoordinateSmoother(buffer_size=SMOOTHING_BUFFER_SIZE)
    smoother_right = CoordinateSmoother(buffer_size=SMOOTHING_BUFFER_SIZE)

    # Persist last known positions so hands don't snap to 0 when tracking is lost
    pos_L = (0, 0, 0)
    pos_R = (0, 0, 0)

    print("\nTracking started. Press 'q' to quit.")

    while True:
        try:
            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()

            if not ret_l or not ret_r:
                time.sleep(0.01)
                continue

            # Rectify
            rect_l = cv2.remap(frame_l, map_l_x, map_l_y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map_r_x, map_r_y, cv2.INTER_LINEAR)

            # --- PROCESS LEFT HAND ---
            new_L = process_hand(rect_l, rect_r, HSV_LEFT_LOWER, HSV_LEFT_UPPER, P1, P2, smoother_left,
                                 CAMERA_TILT_ANGLE)
            if new_L: pos_L = new_L

            # --- PROCESS RIGHT HAND ---
            new_R = process_hand(rect_l, rect_r, HSV_RIGHT_LOWER, HSV_RIGHT_UPPER, P1, P2, smoother_right,
                                 CAMERA_TILT_ANGLE)
            if new_R: pos_R = new_R

            # --- SEND DATA ---
            # Format: "Lx,Ly,Lz|Rx,Ry,Rz"
            msg = f"{pos_L[0]:.2f},{pos_L[1]:.2f},{pos_L[2]:.2f}|{pos_R[0]:.2f},{pos_R[1]:.2f},{pos_R[2]:.2f}"
            sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
            print(msg)

            # --- VISUALIZE ---
            # Visualizer currently only supports one point, so we just show Left hand for now
            # You can update visualizer later to accept two tuples
            vis_frame = draw_visualizer(pos_L[0], pos_L[1], pos_L[2])

            cv2.imshow('3D Data', vis_frame)
            cv2.imshow('Left Eye', rect_l)
            cv2.imshow('Right Eye', rect_r)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)

    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()