import cv2  # This module is used for image processing
import numpy as np  # This module is used for numerical operations
import os  # This module is used for file operations
import torch  # This module is used for deep learning
from lightglue import LightGlue, SuperPoint  # These modules are used for feature extraction
from lightglue.utils import rbd  # Utility function for removing batch dimension
import matplotlib.pyplot as plt  # This module is used for plotting

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if a GPU is available

# SuperPoint+LightGlue with custom parameters
extractor = SuperPoint(
    max_num_keypoints=10000,  # Maximum number of keypoints
    detection_threshold=0.2,  # Detection threshold
    nms_radius=15  # Non-maximum suppression radius
).eval().to(device)

matcher = LightGlue(
    features='superpoint',
    filter_threshold=0.8  # Custom filter threshold
).eval().to(device)

def normalize_image(image):
    """Normalize the image to [0,1] and convert to torch tensor."""
    return torch.tensor(image / 255.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)  # Normalize the image and convert to tensor

def crop_image(image):
    """Crop the top and bottom 15% of the image."""
    height = image.shape[0]
    cropped_image = image[int(height * 0.15):int(height * 0.85), :]  # Crop the image
    return cropped_image

def detect_and_compute_superpoint(image):
    """Detect and compute SuperPoint features."""
    cropped_image = crop_image(image)  # Crop the image
    normalized_image = normalize_image(cropped_image)  # Normalize the cropped image
    feats = extractor.extract(normalized_image)  # Extract features
    return feats

def match_features_superpoint(featsA, featsB):
    """Match features using LightGlue."""
    matches = matcher({'image0': featsA, 'image1': featsB})  # Match features
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]  # Remove batch dimension
    return featsA, featsB, matches['matches']

def custom_nms(keypoints, radius):
    """Custom Non-Maximum Suppression (NMS) to control keypoint distribution."""
    if len(keypoints) == 0:
        return keypoints
    keypoints = sorted(keypoints, key=lambda x: -x.response)  # This line sorts the keypoints by response
    keep = []
    for kp in keypoints:
        if all(np.linalg.norm(np.array(kp.pt) - np.array(keep_kp.pt)) >= radius for keep_kp in keep):
            keep.append(kp)  # Keep keypoints that are not within the NMS radius
    return keep

class UAVNavigator:
    def __init__(self, gps_per_pixel_x, gps_per_pixel_y):
        self.gps_per_pixel_x = gps_per_pixel_x  # Store GPS conversion factor for longitude
        self.gps_per_pixel_y = gps_per_pixel_y  # Store GPS conversion factor for latitude
        
        # Lists to store images, descriptors, keypoints, and GPS coordinates
        self.stored_images = []
        self.stored_features = []
        self.stored_gps = []

    def add_image(self, image, gps_coordinates):
        """Add an image and its GPS coordinates to the stored list."""
        features = detect_and_compute_superpoint(image)  # Detect and compute features
        self.stored_images.append(image)  # Store the original image
        self.stored_features.append(features)  # Store the features
        self.stored_gps.append(gps_coordinates)  # Store the GPS coordinates

    def _compute_homography(self, feats1, feats2):
        """Compute the homography matrix using RANSAC from matched features."""
        feats1, feats2, matches = match_features_superpoint(feats1, feats2)  # Match features
        
        keypoints1 = feats1['keypoints'].cpu().numpy()  # Convert keypoints to numpy array
        keypoints2 = feats2['keypoints'].cpu().numpy()  # Convert keypoints to numpy array
        matches = matches.cpu().numpy()  # Convert matches to numpy array

        if len(matches) < 4:
            return None

        src_pts = keypoints1[matches[:, 0]].reshape(-1, 1, 2)  # Source points for homography
        dst_pts = keypoints2[matches[:, 1]].reshape(-1, 1, 2)  # Destination points for homography

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # Compute homography matrix
        return M, mask, matches

    def infer_current_gps(self, current_image, current_index):
        """Infer the current GPS location based on the highest correlated stored image."""
        current_features = detect_and_compute_superpoint(current_image)  # Detect and compute features

        max_matches = 0
        best_homography = None
        best_index = -1

        # Only compare with prior images
        for i in range(current_index):
            result = self._compute_homography(current_features, self.stored_features[i])
            if result is None:
                continue

            M, mask, matches = result
            good_matches = np.sum(mask)  # Count the number of good matches

            if good_matches > max_matches:
                max_matches = good_matches
                best_homography = M
                best_index = i

        if best_homography is not None and best_index != -1:
            h, w = self.stored_images[best_index].shape[:2]
            center_pt = np.array([[w / 2, h / 2]], dtype='float32').reshape(-1, 1, 2)  # Center point of the image
            transformed_center = cv2.perspectiveTransform(center_pt, best_homography)  # Transform center point
            delta_x, delta_y = transformed_center[0][0] - center_pt[0][0]  # Calculate the shift in x and y

            gps_x, gps_y = self.stored_gps[best_index]
            current_gps_x = gps_x + delta_x * self.gps_per_pixel_x  # Calculate the current GPS longitude
            current_gps_y = gps_y + delta_y * self.gps_per_pixel_y  # Calculate the current GPS latitude

            return best_index, (current_gps_x, current_gps_y)
        return None, None

def parse_gps(file_path):
    """Parse GPS coordinates from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lat_str = lines[0].strip()
        lon_str = lines[1].strip()
        
        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_str)  # Parse latitude
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_str)  # Parse longitude

        lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)  # Convert latitude to decimal
        lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)  # Convert longitude to decimal

        return lat, lon

def parse_dms(dms_str):
    """Parse degrees, minutes, seconds from a DMS string."""
    dms_str = dms_str.replace('°', ' ').replace('\'', ' ').replace('"', ' ')  # Replace degree, minute, second symbols
    parts = dms_str.split()
    deg = int(parts[0])  # Degrees
    min = int(parts[1])  # Minutes
    sec = float(parts[2])  # Seconds
    dir = parts[3]  # Direction (N, S, E, W)
    return deg, min, sec, dir

def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0  # Convert to decimal
    if dir in ['S', 'W']:
        decimal = -decimal  # Negative for South and West
    return decimal

def main():
    gps_per_pixel_x = 0.703e-4  # Longitude
    gps_per_pixel_y = 2.703e-4  # Latitude
    navigator = UAVNavigator(gps_per_pixel_x, gps_per_pixel_y)  # Initialize the UAVNavigator

    # Directory containing the images and GPS files
    directory = './GoogleEarth/SET1'

    # Lists to store actual and estimated GPS coordinates
    actual_gps_list = []
    estimated_gps_list = []

    # Add images and GPS coordinates to the navigator
    for i in range(1, 7): # the difference is equivalent to num pics
        image_path = os.path.join(directory, f'{i}.jpg')
        gps_path = os.path.join(directory, f'{i}.txt')
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
        gps_coordinates = parse_gps(gps_path)  # Parse the GPS coordinates
        
        navigator.add_image(image, gps_coordinates)  # Add the image and GPS coordinates to the navigator

    # Simulate backward flight (GPS lost)
    for i in range(5, 0, -1): # the difference is equivalent to num pics
        current_image = navigator.stored_images[i]
        best_index, estimated_gps = navigator.infer_current_gps(current_image, i)

        if best_index is not None and estimated_gps is not None:
            actual_gps = navigator.stored_gps[i]
            actual_gps_list.append(actual_gps)
            estimated_gps_list.append(estimated_gps)
            print(f"Image {i+1} best match with the following: Image {best_index+1}")
            print(f"Estimated GPS Location for image {i+1}: {estimated_gps}")
            print(f"Actual GPS Location for image {i+1}: {actual_gps}")
            print(f"Deviation-x (%) = {abs(estimated_gps[0] - actual_gps[0]) / abs(actual_gps[0]) * 100}")
            print(f"Deviation-y (%) = {abs(estimated_gps[1] - actual_gps[1]) / abs(actual_gps[1]) * 100}")   
        else:
            print(f"Image {i+1}: Unable to estimate GPS location")

    # Separate the GPS coordinates into latitude and longitude
    actual_gps_x = [coord[0] for coord in actual_gps_list]
    actual_gps_y = [coord[1] for coord in actual_gps_list]
    estimated_gps_x = [coord[0] for coord in estimated_gps_list]
    estimated_gps_y = [coord[1] for coord in estimated_gps_list]

    # Plot the actual and estimated GPS coordinates for longitude
    plt.figure()
    plt.plot(range(1, len(actual_gps_x)+1), actual_gps_x, label='Actual Longitude')
    plt.plot(range(1, len(estimated_gps_x)+1), estimated_gps_x, label='Estimated Longitude')
    plt.xlabel('Image Index')
    plt.ylabel('Longitude')
    plt.legend()
    plt.title('Actual vs Estimated GPS Longitude')
    plt.show()

    # Plot the actual and estimated GPS coordinates for latitude
    plt.figure()
    plt.plot(range(1, len(actual_gps_y)+1), actual_gps_y, label='Actual Latitude')
    plt.plot(range(1, len(estimated_gps_y)+1), estimated_gps_y, label='Estimated Latitude')
    plt.xlabel('Image Index')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('Actual vs Estimated GPS Latitude')
    plt.show()

if __name__ == "__main__":
    main()  # Run the main function
