import cv2
import numpy as np
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import torch

# Constants
INTERVAL_SECONDS = 2.0  # Sampling at 0.5 frames per second
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SuperPoint+LightGlue with custom parameters
extractor = SuperPoint(
    max_num_keypoints=2048,  # Maximum number of keypoints
    detection_threshold=0.11,  # Detection threshold. 0.11 / 0.005
    nms_radius=5  # Non-maximum suppression radius
).eval().to(device)

matcher = LightGlue(
    features='superpoint',
    filter_threshold=0.995  # Custom filter threshold
).eval().to(device)

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error: Could not open video.")
    return cap

def get_video_properties(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = int(fps * INTERVAL_SECONDS)
    return fps, interval, frame_width, frame_height

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return image

def draw_path(frame, points, color=(0, 0, 255)):
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, color, -1)  # Large circles
    return frame

def track_feature(video_path):
    cap = initialize_video_capture(video_path)
    fps, interval, frame_width, frame_height = get_video_properties(cap)
    
    # Resize the frame for display
    scale_width = WINDOW_WIDTH / frame_width
    scale_height = WINDOW_HEIGHT / frame_height
    scale = min(scale_width, scale_height)
    display_size = (int(frame_width * scale), int(frame_height * scale))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_path.mp4', fourcc, fps / interval, display_size)  # Adjust fps for output video
    
    ret, old_frame = cap.read()
    if not ret:
        raise IOError("Error: Could not read initial frame.")
    
    old_image = process_frame(old_frame)
    prev_feats = extractor.extract(old_image)
    
    trajectory = []
    total_displacement = np.array([0.0, 0.0])
    path_points = []
    accumulated_points = []

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            curr_image = process_frame(frame)
            curr_feats = extractor.extract(curr_image)
            
            if prev_feats is not None:
                try:
                    matches = matcher({'image0': prev_feats, 'image1': curr_feats})

                    # Convert matches to numpy array if they are lists
                    matches_data = matches['matches']
                    if isinstance(matches_data, list):
                        matches_data = np.array(matches_data)
                    
                    # Extract keypoints and matches
                    prev_kpts = prev_feats['keypoints'].squeeze(0).cpu().numpy()
                    curr_kpts = curr_feats['keypoints'].squeeze(0).cpu().numpy()
                    matches_data = matches_data.squeeze(0)

                    # Verify the shapes of keypoints and matches
                    if prev_kpts.shape[1] == 2 and curr_kpts.shape[1] == 2 and matches_data.shape[1] == 2:
                        m_kpts0 = prev_kpts[matches_data[:, 0]]
                        m_kpts1 = curr_kpts[matches_data[:, 1]]

                        # Compute pixel changes
                        pixel_changes = m_kpts1 - m_kpts0

                        if pixel_changes.size > 0:
                            total_displacement += np.mean(pixel_changes, axis=0)
                            trajectory.append(total_displacement.copy())
                            path_points.append((int(total_displacement[0]), int(total_displacement[1])))
                            accumulated_points.extend(m_kpts1.tolist())
                            print(f"Total displacement: {total_displacement}")

                            # Draw accumulated keypoints and matches on the frame
                            for pt in accumulated_points:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)  # Large circles
                    else:
                        print("Keypoints or matches dimensions are incorrect.")
                except Exception as e:
                    print(f"Error during matching: {e}")
                    prev_feats = curr_feats
            else:
                prev_feats = curr_feats

            # Draw the path on the frame
            frame = draw_path(frame, path_points)

        # Resize for display
        frame_display = cv2.resize(frame, display_size)
        
        # Write the frame to the output video
        out.write(frame_display)

        frame_count += 1

    cap.release()
    out.release()

    if trajectory:
        trajectory = np.array(trajectory)

        # Plot the trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-')
        plt.title('UAV Trajectory based on Feature Tracking')
        plt.xlabel('X Displacement (pixels)')
        plt.ylabel('Y Displacement (pixels)')
        plt.grid(True)
        plt.show()
    else:
        print("No trajectory data to plot.")

# Usage
track_feature('./1030fps.mp4')
