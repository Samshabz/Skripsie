import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import torch

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SuperPoint+LightGlue with custom parameters
extractor = SuperPoint(
    max_num_keypoints=10000,  # Maximum number of keypoints
    detection_threshold=0.25,  # Detection threshold
    nms_radius=5  # Non-maximum suppression radius
).eval().to(device)

matcher = LightGlue(
    features='superpoint',
    filter_threshold=0.9  # Custom filter threshold
).eval().to(device)

def detect_and_compute_superpoint(image):
    # Normalize image to [0,1] and convert to torch tensor
    image = torch.tensor(image / 255.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    # Extract local features
    feats = extractor.extract(image)
    return feats

def match_features_superpoint(featsA, featsB):
    # Match the features
    matches = matcher({'image0': featsA, 'image1': featsB})
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]  # remove batch dimension
    return featsA, featsB, matches['matches']

def track_features_in_video(video_path, num_features=100, skip_frames=30):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Initialize variables
    prev_features = None
    prev_gray_frame = None
    feature_tracks = []
    forward_path = []
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if necessary
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and describe features using SuperPoint
        feats = detect_and_compute_superpoint(gray_frame)

        if prev_features is None:
            prev_features = feats
            prev_gray_frame = gray_frame
            frame_count += 1
            continue

        # Match features with the previous frame
        prev_feats, curr_feats, matches = match_features_superpoint(prev_features, feats)

        # Extract matched keypoints
        keypointsA = prev_feats['keypoints'].cpu().numpy()
        keypointsB = curr_feats['keypoints'].cpu().numpy()
        matches = matches.cpu().numpy()
        
        pointsA = keypointsA[matches[:, 0]]
        pointsB = keypointsB[matches[:, 1]]

        # Convert to float32 (required by Lucas-Kanade method)
        pointsA = pointsA.astype(np.float32)
        pointsB = pointsB.astype(np.float32)

        # Calculate optical flow using Lucas-Kanade method
        new_pointsB, st, err = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, pointsA, None, **lk_params)

        # Ensure st is a boolean array for indexing
        st = st.reshape(-1)

        # Select good points
        good_pointsA = pointsA[st == 1]
        good_new_pointsB = new_pointsB[st == 1]

        # Record the forward path
        feature_tracks.append((good_pointsA, good_new_pointsB, frame.copy(), prev_gray_frame))

        # Store the path coordinates
        forward_path.append(good_pointsA)

        # Update the previous frame and features
        prev_gray_frame = gray_frame
        prev_features = feats
        frame_count += 1
        print(f"Processed frame {frame_count}")

    cap.release()
    return feature_tracks, forward_path

def compute_reverse_path(feature_tracks):
    reverse_path = []
    for i in range(len(feature_tracks) - 1, 0, -1):
        pointsA, pointsB, curr_frame, prev_gray_frame = feature_tracks[i]
        reverse_path.append((pointsB, pointsA, curr_frame))
    return reverse_path

def visualize_paths(forward_path, reverse_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_path = []

    for i, (forward, reverse) in enumerate(zip(forward_path, reverse_path)):
        pointsA, new_pointsB, curr_frame = reverse
        new_pointsA, pointsB, prev_frame, prev_gray_frame = forward

        # Superimpose on the current frame
        image_color = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        # Draw initial points and new points for reverse path
        for (a, b), (c, d) in zip(new_pointsA, pointsB):
            image_color = cv2.arrowedLine(image_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2, tipLength=0.5)

        # Save images to files
        plt.figure(figsize=(10, 5))
        plt.title(f'Backtrack Path Frame {i+1}')
        plt.imshow(image_color)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'path_visualization_frame_{i+1}.png'))
        plt.close()

        # Append to combined path
        combined_path.append(pointsB)

    return combined_path

def visualize_combined_path(forward_path, combined_path, output_dir):
    plt.figure(figsize=(10, 10))
    plt.title('Forward and Backward Combined Path')
    
    # Plot forward path
    for points in forward_path:
        plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o', label='Forward Path' if not plt.gca().get_legend_handles_labels()[1] else "")

    # Plot backward path
    for points in combined_path:
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='x', label='Backward Path' if not plt.gca().get_legend_handles_labels()[1] else "")

    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'combined_path.png'))
    plt.close()

def main(video_path):
    feature_tracks, forward_path = track_features_in_video(video_path)
    if feature_tracks is None:
        return

    reverse_path = compute_reverse_path(feature_tracks)

    output_dir = 'pathvis'
    combined_path = visualize_paths(feature_tracks, reverse_path, output_dir)

    visualize_combined_path(forward_path, combined_path, output_dir)

    # Print some summary statistics
    total_frames = len(feature_tracks)
    print(f"Total frames processed: {total_frames}")

    if total_frames > 0:
        avg_features_per_frame = np.mean([len(track[0]) for track in feature_tracks])
        print(f"Average number of features tracked per frame: {avg_features_per_frame}")

if __name__ == "__main__":
    video_path = "./forest.mp4"
    main(video_path)
