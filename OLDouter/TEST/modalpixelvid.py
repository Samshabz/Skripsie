import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the interval in seconds between frames to capture
INTERVALSECONDS = 1

def track_feature(video_path, initial_point):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * INTERVALSECONDS)  # Interval in frames

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Capture the initial frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = np.array([[initial_point]], dtype=np.float32)  # Initial point to track

    # Lists to store tracked points
    tracked_points = [initial_point]

    # Loop over frames at intervals
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + interval)  # Move to the next frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow to get new position
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if st[0][0] == 1:  # If the feature is found
            new_point = p1[0][0]
            tracked_points.append(new_point)
            p0 = p1  # Update the previous points

            # Draw all tracked points
            for idx, point in enumerate(tracked_points):
                color = (0, 0, 255) if idx == 0 else (0, 255, 0)  # Red for the first point, green for others
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, color, -1)
        
        old_gray = frame_gray.copy()

        # Display the frame with the tracked points
        cv2.imshow('Tracked Feature', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Plot the tracked points
    tracked_points = np.array(tracked_points)
    plt.plot(tracked_points[:, 0], tracked_points[:, 1], marker='o')
    plt.title('Tracked Feature Points')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()

# Explanation:
# video_path: Path to the video file.
# initial_point: Initial (x, y) point to start tracking.

# Usage
track_feature('./1030fps.mp4', (320, 240))  # Example initial point
