import cv2
import numpy as np
import time

# --- 1. CONFIGURATION ---
# ... (your hsv_lower, hsv_upper, HIT_ZONE) ...
hsv_lower = np.array([40, 100, 100])
hsv_upper = np.array([80, 255, 255])
HIT_ZONE = (200, 100, 400, 300)


# --- 2. THE EXPANDABLE FUNCTION ---
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
    # Use your .local address
    url = "http://esp32-camera-01.local/stream"
    cap = cv2.VideoCapture(url)

    if cap.isOpened():
        print("Connected to stream. Press 'q' to quit.")
    else:
        print("Warning: Initial connection failed. Will retry...")

    while True:
        try:
            # If connection is lost, enter a blocking reconnect loop
            if not cap.isOpened():
                print("Connection lost. Attempting to reconnect...")
                # Loop until we are reconnected
                while not cap.isOpened():
                    cap.release()  # Fully release
                    cap = cv2.VideoCapture(url)  # Try to reconnect
                    if not cap.isOpened():
                        print("Reconnect failed. Retrying in 3 seconds...")
                        time.sleep(3)  # Wait 3 seconds before trying again
                    else:
                        print("Reconnected successfully!")
                        # Break this inner loop and proceed
                        break

            # Now, try to read the frame
            ret, frame = cap.read()

            if not ret:
                # This means connection was 'open' but read failed (stream blip)
                print("Error: Can't receive frame. Resetting connection...")
                cap.release()  # Force the 'isOpened()' check above to fail next loop
                time.sleep(1)  # Brief pause
                continue  # Go to top of main loop


            # --- Tracking Logic ---

            # Flip the frame horizontally (so it's like a mirror)
            # frame = cv2.flip(frame, 1) # Test if you need this

            # Call our modular function to find the object
            target_data, debug_mask = find_target(frame)

            # --- Hit Detection Logic ---
            if target_data:
                x, y, radius = target_data
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                hz_x1, hz_y1, hz_x2, hz_y2 = HIT_ZONE
                if (x > hz_x1 and x < hz_x2) and (y > hz_y1 and y < hz_y2):
                    print(f"HIT! at ({x}, {y})")
                    cv2.putText(frame, "HIT!", (hz_x1, hz_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- Draw UI & Debug Views ---
            cv2.rectangle(frame, (HIT_ZONE[0], HIT_ZONE[1]), (HIT_ZONE[2], HIT_ZONE[3]), (255, 0, 0), 2)
            cv2.imshow('Live Feed', frame)
            cv2.imshow('Debug Mask', debug_mask)

            # --- End of Tracking Logic ---

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            # Catch any other unexpected network error
            print(f"An unexpected error occurred: {e}")
            print("Resetting connection...")
            cap.release()
            time.sleep(2)  # Wait 2 seconds
            continue

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()