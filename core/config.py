import numpy as np
import os

# ==========================================
# HARDWARE SETTINGS
# ==========================================
# IPs
CAM_IP_1 = "192.168.137.102" # Left Camera when standing in front of the punching bag
CAM_IP_2 = "192.168.137.101" # Right Camera when standing in front of the punching bag

# Calibration Resolutions (UXGA - High Res)
LEFT_CAM_URL_CALIB = f"http://{CAM_IP_1}/stream?framesize=13"
RIGHT_CAM_URL_CALIB = f"http://{CAM_IP_2}/stream?framesize=13"

# Tracking Resolutions (VGA - Fast)
LEFT_CAM_URL_TRACKING = f"http://{CAM_IP_1}/stream?framesize=8"
RIGHT_CAM_URL_TRACKING = f"http://{CAM_IP_2}/stream?framesize=8"

# ==========================================
# TRACKING SETTINGS
# ==========================================

# 1. Define the Data Directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# 2. Define the File Path
CALIB_FILE = os.path.join(DATA_DIR, "stereo_calib.yml")

# Target Color (Pink/Magenta)
HSV_LOWER = np.array([35, 42, 119])
HSV_UPPER = np.array([66, 255, 255])

# Geometry
CAMERA_TILT_ANGLE = 40

# Smoothing
SMOOTHING_BUFFER_SIZE = 5

# Networking
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# ==========================================
# VISUALIZER RANGES
# ==========================================
VIS_RANGES = {
    'x_min': 0.15, 'x_max': 0.7,
    'y_min': 0.6, 'y_max': 1.0,
    'z_min': -0.5, 'z_max': -0.15
}