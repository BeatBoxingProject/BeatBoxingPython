import cv2
import time
import socket
import numpy as np
from core.config import *
from core.camera import CameraStream
from core.vision import StereoCamera, CoordinateSmoother
from core.visualizer import draw_visualizer


def main():
    # 1. Init Network
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 2. Init Cameras
    print("Starting Cameras...")
    cam_l = CameraStream(LEFT_CAM_URL_TRACKING, "Left")
    cam_r = CameraStream(RIGHT_CAM_URL_TRACKING, "Right")

    # Wait for first frame to get resolution
    while not cam_l.ret: time.sleep(0.1)
    h, w = cam_l.frame.shape[:2]

    # 3. Init Stereo Logic
    print(f"Loading Calibration ({w}x{h})...")
    stereo = StereoCamera(CALIB_FILE, w, h, CAMERA_TILT_ANGLE)

    smooth_l = CoordinateSmoother(SMOOTHING_BUFFER_SIZE)
    smooth_r = CoordinateSmoother(SMOOTHING_BUFFER_SIZE)

    # Last known valid positions (for UDP persistence)
    udp_pos_L = (0, 0, 0)
    udp_pos_R = (0, 0, 0)

    print("\n🚀 Tracking System Active. Press 'q' to quit.")

    while True:
        if not cam_l.ret or not cam_r.ret: continue

        # Use RAW frames directly
        raw_l, raw_r = cam_l.frame, cam_r.frame

        # Pass RAW frames to get_position (it handles the math internally)
        curr_L = stereo.get_position(raw_l, raw_r, HSV_LEFT_LOWER, HSV_LEFT_UPPER)
        curr_R = stereo.get_position(raw_l, raw_r, HSV_RIGHT_LOWER, HSV_RIGHT_UPPER)

        # C. Smooth & Persist Data
        if curr_L: udp_pos_L = smooth_l.update(*curr_L)
        if curr_R: udp_pos_R = smooth_r.update(*curr_R)

        # D. Send UDP (Always send last known valid data to Unity)
        msg = f"{udp_pos_L[0]:.2f},{udp_pos_L[1]:.2f},{udp_pos_L[2]:.2f}|" \
              f"{udp_pos_R[0]:.2f},{udp_pos_R[1]:.2f},{udp_pos_R[2]:.2f}"
        sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

        # Visualize RAW frames (looks normal!)
        vis_frame = draw_visualizer(curr_L if curr_L else None, curr_R if curr_R else None)

        debug_view = np.hstack((raw_l, raw_r))
        debug_view = cv2.resize(debug_view, (800, 300))

        cv2.imshow("Stereo Tracking (Raw View)", debug_view)
        cv2.imshow("3D Data", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()