    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt

    # Open the video file
    cap = cv.VideoCapture('./forest.mp4')
    latest_off = None
    # Get the frame rate of the video
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps!=0:
        time_between_frames = 1.0 / fps  # Time between each frame
    else:
        time_between_frames = 0.03

    # Define the desired sampling rate in frames per second (e.g., 10 FPS)
    desired_fps = 10
    desired_interval = 1.0 / desired_fps  # Time interval for desired FPS

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100,  # Maximum number of corners to detect
                        qualityLevel=0.3,  # Minimum accepted quality of corners
                        minDistance=7,  # Minimum possible Euclidean distance between the returned corners
                        blockSize=7)  # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),  # Size of the search window at each pyramid level
                    maxLevel=2,  # Maximum number of pyramid levels including the initial image
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))  # Termination criteria of the iterative search algorithm

    # Create some random colors for visualization
    color = np.random.randint(0, 255, (100, 3))

    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        exit()

    # Convert the first frame to grayscale
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    # Detect initial feature points
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Define the scaling factor for resizing the frame (e.g., 0.5 for half size)
    scaling_factor = 0.4

    # Initialize variables for tracking displacement and frame count
    total_displacement = 0
    total_frames = 0
    displacement_list = []

    # Initialize variable for average path of features
    average_path = []

    # Initialize the timer
    current_time = 0.0

    # Margin for edge detection
    edge_margin = 20

    def remove_features_near_edge(features, frame_shape, margin):
        h, w = frame_shape[:2]
        valid_indices = [i for i, pt in enumerate(features) if margin < pt[0] < w - margin and margin < pt[1] < h - margin]
        return np.array([features[i] for i in valid_indices])

    firstrun = True


    def detect_new_features(gray_frame, existing_features, feature_params, min_features=30):
        global latest_off  # Access the global latest_off variable
        if existing_features is not None and len(existing_features) < min_features:
            new_features = cv.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
            if new_features is not None:
                if existing_features is None:
                    return new_features
                else:
                    if len(existing_features) > 0 and latest_off is not None:
                        # Compute offset for new features based on the difference between the first new feature and the current offset
                        new_offset = new_features[0][0] - existing_features[0][0]
                        latest_off += new_offset
                    existing_features = np.concatenate((existing_features, new_features), axis=0)
        return existing_features

    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        current_time += time_between_frames
        if current_time < desired_interval:
            continue
        current_time = 0.0

        # Convert the current frame to grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Remove features near the edge
        if p0 is not None:
            p0 = remove_features_near_edge(p0.reshape(-1, 2), frame.shape, edge_margin).reshape(-1, 1, 2)
            if len(p0) == 0:
                p0 = None

        # Detect new features if the number of valid features falls below 30
        p0 = detect_new_features(frame_gray, p0, feature_params)
        if firstrun:
            latest_off = p0[0][0] if p0 is not None else np.zeros(2)
            firstrun = False
        
        # Update the colors array to match the number of features
        if p0 is not None and len(p0) > len(color):
            additional_colors = np.random.randint(0, 255, (len(p0) - len(color), 3))
            color = np.vstack((color, additional_colors))

        # Only calculate optical flow if there are valid points
        if p0 is not None and len(p0) > 0:
            # Calculate optical flow to track the features
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]  # New positions of the features
                good_old = p0[st == 1]  # Old positions of the features
                good_new_dup = good_new - latest_off
                good_old_dup = good_new - latest_off
                # Calculate displacement and accumulate
                displacements = []
                for (new, old) in zip(good_new, good_old):
                    displacement = np.linalg.norm(new - old)  # Calculate Euclidean distance between new and old positions
                    displacements.append(displacement)

                frame_displacement = np.mean(displacements)  # Average displacement in the current frame
                total_displacement += frame_displacement  # Accumulate total displacement
                displacement_list.append(frame_displacement)  # Store frame displacement
                total_frames += 1  # Increment frame count

                # Calculate and store the average position of the good points (x and y individually)
                avg_position = np.mean(good_new_dup, axis=0)
                if not np.isnan(avg_position).any():  # Check for NaN values
                    average_path.append(avg_position)  # Append valid positions only

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # New position
                c, d = old.ravel()  # Old position
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)  # Draw line
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)  # Draw circle
            img = cv.add(frame, mask)  # Add mask to frame

            # Resize the image for display
            img_resized = cv.resize(img, (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))

            # Display the frame
            cv.imshow('frame', img_resized)
            k = cv.waitKey(30) & 0xff
            if k == 27:  # Exit if 'ESC' key is pressed
                break

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    # Release all windows
    cv.destroyAllWindows()

    # Calculate average speed and variance
    average_speed = total_displacement / (total_frames * desired_interval)
    displacement_variance = np.var(displacement_list)
    speed_variance = np.var(np.array(displacement_list) / desired_interval)

    # Print statistics
    print(f'Total Displacement: {total_displacement:.2f} pixels')
    print(f'Average Speed: {average_speed:.2f} pixels/second')
    print(f'Displacement Variance: {displacement_variance:.2f} pixels^2')
    print(f'Speed Variance: {speed_variance:.2f} (pixels/second)^2')

    # Convert the average path to a numpy array
    average_path = np.array(average_path)

    def reversepath(path):
        # Initialize the UAV's forward path at the origin
        uav_forward_path = [np.array([0, 0])]
        
        # Convert feature movement to UAV movement
        for i in range(1, len(path)):
            movement = path[i-1] - path[i]  # Calculate movement vector
            new_position = uav_forward_path[-1] + movement  # Update UAV position
            uav_forward_path.append(new_position)
        
        # Convert to numpy array
        uav_forward_path = np.array(uav_forward_path)
        
        # Reverse the path
        uav_reverse_path = uav_forward_path[::-1]
        
        return uav_forward_path, uav_reverse_path

    # Get the forward and reverse paths for the UAV
    frwd_path, newbackpath = reversepath(average_path)

    # Print paths for debugging
    print("Feature Path:\n", average_path)
    print("UAV Forward Path:\n", frwd_path)
    print("UAV Reverse Path:\n", newbackpath)

    # Combine plots into a single figure
    plt.figure(figsize=(15, 9))

    # Plot the forward path of the UAV
    plt.subplot(1, 3, 1)
    plt.plot(frwd_path[:, 0], frwd_path[:, 1], 'g-', label='UAV Forward Path')
    plt.scatter(frwd_path[:, 0], frwd_path[:, 1], c='b', s=10)
    plt.scatter(frwd_path[0, 0], frwd_path[0, 1], c='r', s=50, label='Start Point')  # Start point in red
    plt.title('UAV Forward Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

    # Plot the feature path
    plt.subplot(1, 3, 2)
    plt.plot(average_path[:, 0], average_path[:, 1], 'g-', label='Feature Path')
    plt.scatter(average_path[:, 0], average_path[:, 1], c='b', s=10)
    plt.scatter(average_path[0, 0], average_path[0, 1], c='r', s=50, label='Start Point')  # Start point in red
    plt.title('Feature Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

    # Plot the reverse path of the UAV
    plt.subplot(1, 3, 3)
    plt.plot(newbackpath[:, 0], newbackpath[:, 1], 'r-', label='UAV Reverse Path')
    plt.scatter(newbackpath[:, 0], newbackpath[:, 1], c='b', s=10)
    plt.scatter(newbackpath[0, 0], newbackpath[0, 1], c='r', s=50, label='Start Point')  # Start point in red
    plt.title('UAV Reverse Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

    plt.tight_layout()
    plt.show()
