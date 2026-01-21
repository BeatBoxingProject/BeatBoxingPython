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
        return (sum(self.x_buffer)/len(self.x_buffer), 
                sum(self.y_buffer)/len(self.y_buffer), 
                sum(self.z_buffer)/len(self.z_buffer))

class StereoCamera:
    """
    Handles all 3D math: Calibration loading, Triangulation, and Tilt Correction.
    """
    def __init__(self, calib_file, width, height, tilt_angle):
        self.tilt_angle = tilt_angle
        self.calib_data = self._load_calibration(calib_file, width, height)
        
        # Unpack essential matrices for fast access
        self.P1 = self.calib_data['P1']
        self.P2 = self.calib_data['P2']
        
        # Create Rectification Maps once
        self.map_l = cv2.initUndistortRectifyMap(
            self.calib_data['cameraMatrix1'], self.calib_data['distCoeffs1'], 
            self.calib_data['R1'], self.calib_data['P1'], (width, height), cv2.CV_32FC1
        )
        self.map_r = cv2.initUndistortRectifyMap(
            self.calib_data['cameraMatrix2'], self.calib_data['distCoeffs2'], 
            self.calib_data['R2'], self.calib_data['P2'], (width, height), cv2.CV_32FC1
        )

    def rectify_frames(self, frame_l, frame_r):
        rect_l = cv2.remap(frame_l, self.map_l[0], self.map_l[1], cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, self.map_r[0], self.map_r[1], cv2.INTER_LINEAR)
        return rect_l, rect_r

    def get_position(self, rect_l, rect_r, hsv_lower, hsv_upper):
        """
        Returns (x, y, z) tuple or None if target not found.
        """
        # 1. Find targets
        target_l = self._find_contour(rect_l, hsv_lower, hsv_upper)
        target_r = self._find_contour(rect_r, hsv_lower, hsv_upper)

        if target_l and target_r:
            # Draw debug circles on the frames directly
            cv2.circle(rect_l, (target_l[0], target_l[1]), int(target_l[2]), (0, 255, 255), 2)
            cv2.circle(rect_r, (target_r[0], target_r[1]), int(target_r[2]), (0, 255, 255), 2)

            # 2. Triangulate
            pts_4d = cv2.triangulatePoints(
                self.P1, self.P2, 
                np.array(target_l[:2], dtype=float), np.array(target_r[:2], dtype=float)
            )
            pts_3d = pts_4d[:3] / pts_4d[3]
            
            # 3. Tilt Correction
            x, y, z = pts_3d.flatten()
            theta = math.radians(self.tilt_angle)
            y_new = y * math.cos(theta) - z * math.sin(theta)
            z_new = y * math.sin(theta) + z * math.cos(theta)

            # 4. Swap Axes for Unity (X=X, Y=Z, Z=Y)
            return (x, z_new, y_new)
            
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
            if radius > 10: return (int(x), int(y), int(radius))
        return None

    def _load_calibration(self, filename, w, h):
        if not os.path.exists(filename): raise FileNotFoundError("Calibration file missing.")
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        data = {}
        for k in ['cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2', 'R1', 'R2', 'P1', 'P2']:
            data[k] = fs.getNode(k).mat()
        
        # Auto-scale if resolution doesn't match calibration
        orig_w = int(fs.getNode('frame_width').real())
        if orig_w != w:
            scale = w / orig_w
            data['P1'] *= scale
            data['P2'] *= scale
            data['cameraMatrix1'] *= scale
            data['cameraMatrix2'] *= scale
        
        fs.release()
        return data