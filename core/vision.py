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

        # Calculate Average
        avg_x = sum(self.x_buffer) / len(self.x_buffer)
        avg_y = sum(self.y_buffer) / len(self.y_buffer)
        avg_z = sum(self.z_buffer) / len(self.z_buffer)

        return avg_x, avg_y, avg_z

def apply_tilt_correction(x, y, z, angle_degrees):
    theta = math.radians(angle_degrees)
    y_new = y * math.cos(theta) - z * math.sin(theta)
    z_new = y * math.sin(theta) + z * math.cos(theta)
    return x, y_new, z_new

def find_target(frame, hsv_lower, hsv_upper, min_radius=10):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_target = None
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > min_radius:
            found_target = (int(x), int(y), int(radius))
            
    return found_target, mask

def load_calibration_data(filename, target_width, target_height):
    if not os.path.exists(filename):
        print(f"Error: Calibration file not found at {filename}")
        return None
        
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    calib_data = {}
    keys = ['cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2',
            'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q', 'frame_width', 'frame_height']
            
    for key in keys:
        node = fs.getNode(key)
        if node.isNone(): continue
        calib_data[key] = node.mat() if key not in ['frame_width', 'frame_height'] else int(node.real())
    fs.release()

    # Scale calibration if resolution changed
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
        # Update stored size
        calib_data['frame_width'] = target_width
        calib_data['frame_height'] = target_height
        
    return calib_data