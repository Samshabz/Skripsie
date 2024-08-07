import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Define the path to images
image_paths = ['./GoogleEarth/SET1/1.jpg', './GoogleEarth/SET1/2.jpg', './GoogleEarth/SET1/3.jpg', './GoogleEarth/SET1/4.jpg']

# Pixel to kilometer ratio
pixel_to_km_ratio = 1144 / 296

# GPS coordinates of the baseline (frame 1)
baseline_gps = (-31.618886, 21.144586)

# Function to calculate GPS change
def pixel_to_gps_change(pixel_change, ratio, baseline_gps):
    km_change = pixel_change * ratio / 1000  # Convert to kilometers
    # Approximate conversion from km to latitude and longitude degrees
    new_lat = baseline_gps[0] + (km_change / 111)  # Latitude degrees
    new_lon = baseline_gps[1] + (km_change / (111 * np.cos(np.deg2rad(baseline_gps[0]))))  # Longitude degrees
    return (new_lat, new_lon)

# Read and process each image
images = [cv.imread(img_path) for img_path in image_paths]

# Convert the first image to grayscale
old_gray = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)

# Initialize variables for tracking displacement and frame count
total_displacement = 0
displacement_list = []

# Initialize variable for average path of features
average_path = []

# Function to calculate the best match between frames using feature matching
def find_best_match(old_gray, new_gray):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(old_gray, None)
    kp2, des2 = orb.detectAndCompute(new_gray, None)
    # Create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    # Extract the matched points
    pts_old = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts_new = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return kp1, kp2, matches, pts_old, pts_new

# Process each pair of images
for i in range(1, len(images)):
    frame_gray = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)
    
    # Find the best match points
    kp1, kp2, matches, pts_old, pts_new = find_best_match(old_gray, frame_gray)
    
    # Calculate displacement
    displacements = np.linalg.norm(pts_new - pts_old, axis=1)
    frame_displacement = np.mean(displacements)  # Average displacement in the current frame
    total_displacement += frame_displacement  # Accumulate total displacement
    displacement_list.append(frame_displacement)  # Store frame displacement
    
    # Calculate GPS change
    new_gps = pixel_to_gps_change(frame_displacement, pixel_to_km_ratio, baseline_gps)
    baseline_gps = new_gps  # Update the baseline GPS for the next frame
    
    # Print GPS for current frame
    print(f'GPS for frame {i+1}: {new_gps}')
    
    # Draw the matches
    img_matches = cv.drawMatches(images[i-1], kp1, images[i], kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matches)
    plt.show()
    
    # Update the previous frame
    old_gray = frame_gray.copy()

cv.destroyAllWindows()

# Convert the average path to a numpy array
average_path = np.array(displacement_list)

# Function to calculate the forward and reverse paths
def reversepath(path):
    uav_forward_path = [np.array([0, 0])]
    for i in range(1, len(path)):
        movement = path[i-1] - path[i]
        new_position = uav_forward_path[-1] + movement
        uav_forward_path.append(new_position)
    uav_forward_path = np.array(uav_forward_path)
    uav_reverse_path = uav_forward_path[::-1]
    return uav_forward_path, uav_reverse_path

# Get the forward and reverse paths for the UAV
frwd_path, newbackpath = reversepath(average_path)

# Plot the paths
plt.figure(figsize=(15, 5))

# Plot the displacements
plt.subplot(1, 1, 1)
plt.plot(range(1, len(displacement_list)+1), displacement_list, 'g-', label='Displacement')
plt.scatter(range(1, len(displacement_list)+1), displacement_list, c='b', s=10)
plt.title('Frame Displacements')
plt.xlabel('Frame Number')
plt.ylabel('Displacement (pixels)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
