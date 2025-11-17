import cv2
import numpy as np
import time  # We might need this for a small delay

# --- 1. CONFIGURATION ---

# TODO: FIND YOUR COLOR!
hsv_lower = np.array([40, 100, 100])
hsv_upper = np.array([80, 255, 255])

# Define a 2D "Hit Zone"
HIT_ZONE = (200, 100, 400, 300)  # (Top-left X, Top-left Y, Bottom-right X, Bottom-right Y)


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
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find all the "contours" (outlines) of the white blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_target = None

    if len(contours) > 0:
        # Find the biggest contour by area
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            found_target = (int(x), int(y), int(radius))

    return found_target, mask


# --- 3. THE MAIN LOOP ---
# noinspection PyChainedComparisons
def main():
    url = "http://192.168.137.169/stream"  # ESP32 IP
    cap = cv2.VideoCapture(url)

    # Using USB Webcam instead of ESP Stream:
    # cap = cv2.VideoCapture(0) # 0 = internal, 1 = webcam

    if not cap.isOpened():
        print("Error: Cannot open camera on initial attempt.")
        # We can still try to connect in the main loop
    else:
        print("Connected to stream. Press 'q' to quit.")

    while True:
        try:
            # Read one frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Error: Can't receive frame. Attempting to reconnect...")

                # --- This is the new reconnect logic ---
                cap.release()  # Release the broken connection
                cap = cv2.VideoCapture(url)  # Re-establish connection
                time.sleep(1)  # Give it a second to connect
                # ----------------------------------------

                continue  # Skip the rest of this loop iteration

            # Flip the frame horizontally (so it's like a mirror)
            # You might not want this for an ESP stream, test it.
            # frame = cv2.flip(frame, 1)

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

        except Exception as e:
            # Catch any other weird network errors
            print(f"An unexpected error occurred: {e}")
            print("Attempting to reconnect...")
            cap.release()
            cap = cv2.VideoCapture(url)
            time.sleep(2)  # Give it a couple of seconds on a major error

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()