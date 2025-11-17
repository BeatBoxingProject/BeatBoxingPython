import cv2
import numpy as np
import time
import os

# ====================================================================
# --- 1. CONFIGURATION ---
# ====================================================================

# --- Camera URLs ---
url_1 = "http://192.168.137.102/stream"  # Left Camera
url_2 = "http://192.168.137.101/stream"  # Right Camera

# --- Calibration File ---
CALIB_FILE = "stereo_calib.yml"

# --- Target Color (from your original script) ---
# This is for your boxing glove or target
hsv_lower = np.array([40, 100, 100])
hsv_upper = np.array([80, 255, 255])


# ====================================================================
# --- 2. HELPER FUNCTIONS ---
# ====================================================================

def load_calibration_data(filename):
    """
    Loads all matrices from the .yml calibration file.
    """
    if not os.path.exists(filename):
        print(f"Error: Calibration file not found at {filename}")
        return None

    print(f"Loading calibration data from {filename}...")
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"Error: Could not open {filename}")
        return None

    calib_data = {}
    # Read all the necessary matrices
    keys = [
        'cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2',
        'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q',
        'frame_width', 'frame_height'
    ]

    for key in keys:
        calib_data[key] = fs.getNode(key).mat() if key not in ['frame_width', 'frame_height'] else fs.getNode(
            key).real()
        if calib_data[key] is None:
            print(f"Error: Failed to read '{key}' from calibration file.")
            fs.release()
            return None

    # OpenCV's file storage reads single numbers as 1x1 matrices, fix this
    calib_data['frame_width'] = int(calib_data['frame_width'])
    calib_data['frame_height'] = int(calib_data['frame_height'])

    fs.release()
    print("Calibration data loaded successfully.")
    return calib_data


def connect_to_camera(url, camera_name):
    """
    Tries to connect to a camera stream.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Could not open {camera_name} stream at {url}")
        return None

    print(f"Successfully connected to {camera_name} camera.")
    return cap


def find_target(frame):
    """
    Takes a video frame, finds the largest colored object,
    and returns its center (x, y) coordinates and radius.
    (This is your function from beatboxing.py)
    """
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the color mask
    mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)

    # Optional: Clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find all the "contours" (outlines) of the white blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_target = None

    if len(contours) > 0:
        # Find the biggest contour by area
        c = max(contours, key=cv2.contourArea)

        # Get the bounding circle of the biggest contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # We only care about it if it's a decent size
        if radius > 10:
            found_target = (int(x), int(y), int(radius))

    # Return the (x, y, radius) and the mask (for debugging)
    return found_target, mask


# ====================================================================
# --- 3. MAIN APPLICATION ---
# ====================================================================

def main():
    # --- STEP 1: LOAD CALIBRATION ---
    calib_data = load_calibration_data(CALIB_FILE)
    if calib_data is None:
        return  # Exit if calibration failed to load

    # Extract all matrices for convenience
    mtx_l, dist_l = calib_data['cameraMatrix1'], calib_data['distCoeffs1']
    mtx_r, dist_r = calib_data['cameraMatrix2'], calib_data['distCoeffs2']
    R1, R2 = calib_data['R1'], calib_data['R2']
    P1, P2 = calib_data['P1'], calib_data['P2']
    frame_shape = (calib_data['frame_width'], calib_data['frame_height'])

    # --- STEP 3: CREATE RECTIFICATION MAPS ---
    # This is done ONCE outside the loop for efficiency
    print("Creating rectification maps...")
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, frame_shape, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, frame_shape, cv2.CV_32FC1)
    print("Maps created.")

    # --- STEP 2: CONNECT TO STREAMS (Initial) ---
    cap_l = None
    cap_r = None

    print("\nStarting main loop. Press 'q' to quit.")

    while True:
        try:
            # --- Robust Reconnection Logic ---
            if cap_l is None or not cap_l.isOpened():
                cap_l = connect_to_camera(url_1, "Left")
                if cap_l is None:
                    print("Retrying Left Camera in 3s...")
                    time.sleep(3)
                    continue

            if cap_r is None or not cap_r.isOpened():
                cap_r = connect_to_camera(url_2, "Right")
                if cap_r is None:
                    print("Retrying Right Camera in 3s...")
                    time.sleep(3)
                    continue

            # --- Read Frames ---
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            # --- Handle Read Errors ---
            if not ret_l or not ret_r:
                print("Error: Failed to grab frame from one or both cameras.")
                if not ret_l:
                    cap_l.release()
                    cap_l = None
                if not ret_r:
                    cap_r.release()
                    cap_r = None
                time.sleep(1)  # Brief pause before retry
                continue

            # --- STEP 3: RECTIFY FRAMES (Apply Maps) ---
            rect_l = cv2.remap(frame_l, map_l_x, map_l_y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map_r_x, map_r_y, cv2.INTER_LINEAR)

            # --- STEP 4: FIND TARGETS IN BOTH FRAMES ---
            # We use the exact same find_target function as before!
            target_l_data, mask_l = find_target(rect_l)
            target_r_data, mask_r = find_target(rect_r)

            # --- STEP 5: TRIANGULATE THE 3D POINT ---
            # This is the new magic!
            if target_l_data and target_r_data:
                # Get 2D points (x, y)
                p1 = (target_l_data[0], target_l_data[1])
                p2 = (target_r_data[0], target_r_data[1])

                # Draw 2D circles on the (rectified) frames for debugging
                cv2.circle(rect_l, p1, int(target_l_data[2]), (0, 255, 0), 2)
                cv2.circle(rect_r, p2, int(target_r_data[2]), (0, 255, 0), 2)

                # Prep points for triangulation
                # Must be 2xN array of floats
                p1_np = np.array([[p1[0]], [p1[1]]], dtype=np.float64)
                p2_np = np.array([[p2[0]], [p2[1]]], dtype=np.float64)

                # Triangulate
                point_4d_hom = cv2.triangulatePoints(P1, P2, p1_np, p2_np)

                # Convert from homogeneous 4D to 3D coordinates
                point_3d = point_4d_hom / point_4d_hom[3]

                # Extract X, Y, Z
                # (The units will be the same as your SQUARE_SIZE_METERS from calibration)
                x_3d = point_3d[0][0]
                y_3d = point_3d[1][0]
                z_3d = point_3d[2][0]

                # --- Display 3D Coords on Screen ---
                # NOTE: (0,0,0) is the center of the LEFT camera
                coords_text = f"X: {x_3d:.2f}m  Y: {y_3d:.2f}m  Z: {z_3d:.2f}m"
                print(coords_text)  # Print to console

                # Draw on the 'Left' debug window
                cv2.putText(rect_l, coords_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- Show Debug Views ---
            cv2.imshow('Rectified Left', rect_l)
            cv2.imshow('Rectified Right', rect_r)
            cv2.imshow('Mask Left', mask_l)
            cv2.imshow('Mask Right', mask_r)

            # --- Quit Key ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

        except Exception as e:
            # Catch any other unexpected network error
            print(f"An unexpected error occurred: {e}")
            print("Resetting connections...")
            if cap_l: cap_l.release()
            if cap_r: cap_r.release()
            cap_l = None
            cap_r = None
            time.sleep(2)  # Wait 2 seconds

    # --- Cleanup ---
    if cap_l: cap_l.release()
    if cap_r: cap_r.release()
    cv2.destroyAllWindows()
    print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()