import cv2
import numpy as np

# --- 1. CONFIGURATION ---

# TODO: FIND YOUR COLOR!
# This is for a bright "lime" green. You MUST change this.
# Use a script to find these values (search "OpenCV HSV color picker").
# H (Hue): 0-179, S (Saturation): 0-255, V (Value): 0-255
hsv_lower = np.array([40, 100, 100])
hsv_upper = np.array([80, 255, 255])

# Define a 2D "Hit Zone" (x_start, y_start, x_end, y_end)
# We'll draw it on the screen.
HIT_ZONE = (200, 100, 400, 300)  # (Top-left X, Top-left Y, Bottom-right X, Bottom-right Y)


# --- 2. THE EXPANDABLE FUNCTION ---
# This function is the core of our logic.
# It finds the object and returns its (x, y) center.
# This is what we will "plug into" the stereo version later.
def find_target(frame):
    """
    Takes a video frame, finds the largest colored object,
    and returns its center (x, y) coordinates.
    """
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the color mask
    mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)

    # Optional: Clean up the mask
    # Erode and dilate to remove small "noise" pixels
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

    # Return the (x, y) center and the mask (for debugging)
    return found_target, mask


# --- 3. THE MAIN LOOP ---
def main():
    # Start capturing from the default webcam (usually '0')
    cap = cv2.VideoCapture(1) # 0 = internal, 1 = webcam

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        # Read one frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting.")
            break

        # Flip the frame horizontally (so it's like a mirror)
        frame = cv2.flip(frame, 1)

        # --- This is the key part ---
        # Call our modular function to find the object
        target_data, debug_mask = find_target(frame)

        # --- Hit Detection Logic ---
        if target_data:
            x, y, radius = target_data

            # Draw the found object
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Center dot

            # Check if the center (x, y) is inside our HIT_ZONE
            hz_x1, hz_y1, hz_x2, hz_y2 = HIT_ZONE
            if (x > hz_x1 and x < hz_x2) and (y > hz_y1 and y < hz_y2):
                # We have a hit!
                print(f"HIT! at ({x}, {y})")
                cv2.putText(frame, "HIT!", (hz_x1, hz_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Draw UI & Debug Views ---
        # Draw the hit zone rectangle
        cv2.rectangle(frame, (HIT_ZONE[0], HIT_ZONE[1]), (HIT_ZONE[2], HIT_ZONE[3]), (255, 0, 0), 2)

        # Show the video feeds
        cv2.imshow('Live Feed', frame)
        cv2.imshow('Debug Mask', debug_mask)  # See what the computer sees!

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()