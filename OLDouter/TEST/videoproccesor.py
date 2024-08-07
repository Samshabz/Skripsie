import cv2
import matplotlib.pyplot as plt
import numpy as np
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet, match_pair, viz2d
from lightglue.utils import load_image, rbd
import torch

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

# Function to process a frame
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return image

# Open video
input_video_path = './1030fps.mp4'
output_video_path = './output_with_path.mp4'
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_interval = int(fps // 10)  # Process every nth frame to achieve ~10 fps or slower

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables
prev_feats = None
trajectory = []
total_displacement = np.array([0.0, 0.0])
path_points = []

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
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
                        print(f"Total displacement: {total_displacement}")
                else:
                    print("Keypoints or matches dimensions are incorrect.")
            except Exception as e:
                print(f"Error during matching: {e}")
                prev_feats = curr_feats
        else:
            prev_feats = curr_feats

    # Draw the path on the frame
    for point in path_points:
        cv2.circle(frame, point, 2, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Frame with Path', frame)

    # Write the frame to the output video
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

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
