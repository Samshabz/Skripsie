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

def load_images(image_paths):
    images = [cv2.imread(img_path) for img_path in image_paths]
    return images

def resize_images(images):
    min_shape = min((img.shape for img in images), key=lambda x: x[0] * x[1])
    resized_images = [cv2.resize(img, (min_shape[1], min_shape[0])) for img in images]
    return resized_images

def process_images(image_paths, num_features=100):
    images = load_images(image_paths)
    images = resize_images(images)

    if len(images) < 2:
        print("Error: Not enough images provided.")
        return None

    # Initialize variables
    prev_features = None
    prev_gray_frame = None
    feature_tracks = []
    forward_path = []

    for frame_count, frame in enumerate(images):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and describe features using SuperPoint
        feats = detect_and_compute_superpoint(gray_frame)

        if prev_features is None:
            prev_features = feats
            prev_gray_frame = gray_frame
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

        # Ensure st is a boolean array for indexing
        st = np.ones(len(pointsA), dtype=bool)

        # Select good points
        good_pointsA = pointsA[st == 1]
        good_new_pointsB = pointsB[st == 1]

        # Record the forward path
        feature_tracks.append((good_pointsA, good_new_pointsB, frame.copy(), prev_gray_frame))

        # Store the path coordinates
        forward_path.append(good_pointsA)

        # Update the previous frame and features
        prev_gray_frame = gray_frame
        prev_features = feats
        print(f"Processed frame {frame_count + 1}")

    return feature_tracks, forward_path, images

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
        pointsB, pointsA, curr_frame = reverse

        # Superimpose on the current frame
        image_color = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        # Draw initial points and new points for reverse path
        for (a, b), (c, d) in zip(pointsA, pointsB):
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

def calculate_gps_displacement(forward_path, pixel_to_km_ratio, baseline_gps):
    displacements = []
    gps_path = [baseline_gps]
    for i in range(1, len(forward_path)):
        pointsA = forward_path[i-1]
        pointsB = forward_path[i]

        # Ensure both points arrays have the same length
        min_len = min(len(pointsA), len(pointsB))
        pointsA = pointsA[:min_len]
        pointsB = pointsB[:min_len]

        if len(pointsA) == 0 or len(pointsB) == 0:
            continue

        displacement = np.mean(np.linalg.norm(pointsB - pointsA, axis=1))
        displacements.append(displacement)

        km_change = displacement * pixel_to_km_ratio / 1000  # Convert to kilometers
        new_lat = baseline_gps[0] + (km_change / 111)  # Approximate conversion from km to latitude degrees
        new_lon = baseline_gps[1] + (km_change / (111 * np.cos(np.deg2rad(baseline_gps[0]))))  # Approximate conversion from km to longitude degrees
        baseline_gps = (new_lat, new_lon)
        gps_path.append(baseline_gps)

    return displacements, gps_path

def visualize_cross_correlation(images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, len(images)):
        img1 = cv2.cvtColor(images[i-1], cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # Use normalized cross-correlation
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Shift the max location to the center of the result
        shift_y = max_loc[1] - result.shape[0] // 2
        shift_x = max_loc[0] - result.shape[1] // 2

        # Visualize the result
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Image 1')
        plt.imshow(cv2.cvtColor(images[i-1], cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Image 2')
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Cross-Correlation')
        plt.imshow(result, cmap='hot')
        plt.scatter(max_loc[0], max_loc[1], c='blue')
        plt.axis('off')

        plt.suptitle(f'Cross-Correlation between Frame {i} and Frame {i+1} (Max: {max_val:.2f})')
        plt.savefig(os.path.join(output_dir, f'cross_correlation_{i}.png'))
        plt.close()

        print(f'Cross-Correlation between Frame {i} and Frame {i+1}: Max value: {max_val:.2f} at location {max_loc}, shift: ({shift_x}, {shift_y})')

def visualize_feature_matching(feature_tracks, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (pointsA, pointsB, curr_frame, prev_frame) in enumerate(feature_tracks):
        # Ensure both frames have the same size and type
        if prev_frame.shape != curr_frame.shape or prev_frame.dtype != curr_frame.dtype:
            print(f"Skipping visualization for frame {i} due to size/type mismatch")
            continue

        image_matches = cv2.hconcat([prev_frame, curr_frame])

        for (x1, y1), (x2, y2) in zip(pointsA, pointsB):
            x2 = x2 + prev_frame.shape[1]  # Offset x2 by the width of the previous frame
            cv2.circle(image_matches, (int(x1), int(y1)), 5, (255, 0, 0), -1)
            cv2.circle(image_matches, (int(x2), int(y2)), 5, (0, 255, 0), -1)
            cv2.line(image_matches, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        plt.figure(figsize=(15, 5))
        plt.title(f'Feature Matching between Frame {i} and Frame {i+1}')
        plt.imshow(cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'feature_matching_{i}.png'))
        plt.close()

def main(image_paths):
    # Define baseline GPS coordinates and pixel to kilometer ratio
    baseline_gps = (-31.618886, 21.144586)
    pixel_to_km_ratio = 1144 / 296

    feature_tracks, forward_path, images = process_images(image_paths)
    if feature_tracks is None:
        return

    reverse_path = compute_reverse_path(feature_tracks)

    output_dir = 'pathvis'
    combined_path = visualize_paths(forward_path, reverse_path, output_dir)

    visualize_combined_path(forward_path, combined_path, output_dir)

    displacements, gps_path = calculate_gps_displacement(forward_path, pixel_to_km_ratio, baseline_gps)

    # Visualize cross-correlation
    visualize_cross_correlation(images, output_dir)

    # Visualize feature matching
    visualize_feature_matching(feature_tracks, output_dir)

    # Print GPS path for each frame
    for i, gps in enumerate(gps_path):
        print(f'GPS for frame {i+1}: {gps}')

    # Print some summary statistics
    total_frames = len(feature_tracks)
    print(f"Total frames processed: {total_frames}")

    if total_frames > 0:
        avg_features_per_frame = np.mean([len(track[0]) for track in feature_tracks])
        print(f"Average number of features tracked per frame: {avg_features_per_frame}")

if __name__ == "__main__":
    image_paths = ["./GoogleEarth/SET1/1.jpg", "./GoogleEarth/SET1/2.jpg", "./GoogleEarth/SET1/3.jpg", "./GoogleEarth/SET1/4.jpg"]
    main(image_paths)
