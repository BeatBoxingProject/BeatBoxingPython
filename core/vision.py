import cv2
import numpy as np
import math
import os
from collections import deque


class CoordinateSmoother:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)
        self.z_buffer = deque(maxlen=buffer_size)

    def update(self, x, y, z):
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        self.z_buffer.append(z)
        return (sum(self.x_buffer) / len(self.x_buffer),
                sum(self.y_buffer) / len(self.y_buffer),
                sum(self.z_buffer) / len(self.z_buffer))


class StereoCamera:
    """
    Handles 3D math using Point-Based Rectification.
    We do NOT warp the images. We only warp the coordinates.
    """

    def __init__(self, calib_file, width, height, tilt_angle):
        self.tilt_angle = tilt_angle
        self.calib_data = self._load_calibration(calib_file, width, height)

        # We need the RAW intrinsics for undistortion
        self.K1 = self.calib_data['cameraMatrix1']
        self.D1 = self.calib_data['distCoeffs1']
        self.K2 = self.calib_data['cameraMatrix2']
        self.D2 = self.calib_data['distCoeffs2']

        # We need the Rectification transforms for the points
        self.R1 = self.calib_data['R1']
        self.R2 = self.calib_data['R2']
        self.P1 = self.calib_data['P1']
        self.P2 = self.calib_data['P2']

    def get_position(self, raw_l, raw_r, hsv_lower, hsv_upper):
        """
        Takes RAW (curved) frames. Returns (x, y, z) in meters.
        """
        # 1. Find targets in the RAW image (No rectification visual artifacts!)
        target_l = self._find_contour(raw_l, hsv_lower, hsv_upper)
        target_r = self._find_contour(raw_r, hsv_lower, hsv_upper)

        if target_l and target_r:
            # Draw on the raw frames so you can see what's happening
            cv2.circle(raw_l, (target_l[0], target_l[1]), int(target_l[2]), (0, 255, 0), 2)
            cv2.circle(raw_r, (target_r[0], target_r[1]), int(target_r[2]), (0, 255, 0), 2)

            # 2. Undistort & Rectify just these specific points
            # Format must be shape (N, 1, 2) for undistortPoints
            pt_l = np.array([[[target_l[0], target_l[1]]]], dtype=np.float64)
            pt_r = np.array([[[target_r[0], target_r[1]]]], dtype=np.float64)

            # This transforms the raw coordinate into the "Ideal Rectified" coordinate
            rect_pt_l = cv2.undistortPoints(pt_l, self.K1, self.D1, R=self.R1, P=self.P1)
            rect_pt_r = cv2.undistortPoints(pt_r, self.K2, self.D2, R=self.R2, P=self.P2)

            # 3. Triangulate using the Rectified Points and Rectified Projection Matrices
            # (Output of undistortPoints is already shaped (N, 1, 2), we need (2, N))
            pts_4d = cv2.triangulatePoints(
                self.P1, self.P2,
                rect_pt_l.reshape(2, 1),
                rect_pt_r.reshape(2, 1)
            )

            # Convert Homogeneous (4D) to Euclidean (3D)
            pts_3d = pts_4d[:3] / pts_4d[3]
            x, y, z = pts_3d.flatten()

            # 4. Tilt Correction
            theta = math.radians(self.tilt_angle)
            y_new = y * math.cos(theta) - z * math.sin(theta)
            z_new = y * math.sin(theta) + z * math.cos(theta)

            # 5. Swap Axes:
            # If cameras are in front: X->X, Y->Depth(Z), Z->Height(-Y)
            return (x, -z_new, y_new)

        return None

    def _find_contour(self, img, lower, upper):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 5: return (int(x), int(y), int(radius))  # Lowered radius threshold
        return None

    def _load_calibration(self, filename, w, h):
        if not os.path.exists(filename): raise FileNotFoundError("Calibration file missing.")
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        data = {}
        for k in ['cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2', 'R1', 'R2', 'P1', 'P2']:
            data[k] = fs.getNode(k).mat()

        orig_w = int(fs.getNode('frame_width').real())
        if orig_w != w:
            scale = w / orig_w
            for k in ['cameraMatrix1', 'cameraMatrix2', 'P1', 'P2']:
                data[k] *= scale

        fs.release()
        return data