import cv2  # This module is used for image processing
import numpy as np  # This module is used for numerical operations
import os  # This module is used for file operations
import torch  # This module is used for deep learning
from lightglue import LightGlue, SuperPoint  # These modules are used for feature extraction
from lightglue.utils import rbd  # Utility function for removing batch dimension
import matplotlib.pyplot as plt  # This module is used for plotting
from scipy.optimize import minimize  # For optimization

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if a GPU is available

# SuperPoint+LightGlue with custom parameters
extractor = SuperPoint(
    max_num_keypoints=50000,  # Maximum number of keypoints. More keypoints may improve matching but increase computation. It can also increase the chance of false matches / noise. 
    detection_threshold=0.003,  # Detection threshold. Lower values may detect more keypoints but increase noise.
    nms_radius=15  # Non-maximum suppression radius. Higher values may reduce the number of keypoints and improve matching (by reducing noisy duplicate points).
).eval().to(device) # Set the model to evaluation mode and move it to the device

matcher = LightGlue( # Initialize LightGlue
    features='superpoint', # Use SuperPoint features
    filter_threshold=0.1  # Custom filter threshold. Lower values may increase the number of matches but may also increase noise.
).eval().to(device)

def normalize_image(image):
    """Normalize the image to [0,1] and convert to torch tensor."""
    return torch.tensor(image / 255.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)

def crop_image(image):
    """Crop the top and bottom 15% of the image."""
    height = image.shape[0]
    cropped_image = image[int(height * 0.25):int(height * 0.75), :]
    return cropped_image

def detect_and_compute_superpoint(image):
    """Detect and compute SuperPoint features."""
    cropped_image = crop_image(image)
    normalized_image = normalize_image(cropped_image)
    feats = extractor.extract(normalized_image)
    return feats

def match_features_superpoint(featsA, featsB):
    """Match features using LightGlue."""
    matches = matcher({'image0': featsA, 'image1': featsB})
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]
    return featsA, featsB, matches['matches']

def custom_nms(keypoints, radius):
    """Custom Non-Maximum Suppression (NMS) to control keypoint distribution."""
    if len(keypoints) == 0:
        return keypoints
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keep = []
    for kp in keypoints:
        if all(np.linalg.norm(np.array(kp.pt) - np.array(keep_kp.pt)) >= radius for keep_kp in keep):
            keep.append(kp)
    return keep

class UAVNavigator:
    def __init__(self, gps_per_pixel_x, gps_per_pixel_y):
        self.gps_per_pixel_x = gps_per_pixel_x
        self.gps_per_pixel_y = gps_per_pixel_y
        self.stored_images = []
        self.stored_features = []
        self.stored_gps = []

    def add_image(self, image, gps_coordinates):
        """Add an image and its GPS coordinates to the stored list."""
        features = detect_and_compute_superpoint(image)
        self.stored_images.append(image)
        self.stored_features.append(features)
        self.stored_gps.append(gps_coordinates)

    def _compute_homography(self, feats1, feats2):
        """Compute the homography matrix using RANSAC from matched features."""
        feats1, feats2, matches = match_features_superpoint(feats1, feats2)
        
        keypoints1 = feats1['keypoints'].cpu().numpy()
        keypoints2 = feats2['keypoints'].cpu().numpy()
        matches = matches.cpu().numpy()

        if len(matches) < 4:
            return None

        src_pts = keypoints1[matches[:, 0]].reshape(-1, 1, 2)
        dst_pts = keypoints2[matches[:, 1]].reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M, mask, matches

    def calculate_reprojection_error(self, src_pts, dst_pts, M):
        """Calculate the reprojection error of the homography matrix."""
        if M is None:
            return float('inf')  # Infinite error if no homography can be computed

        projected_pts = cv2.perspectiveTransform(src_pts, M)
        errors = np.linalg.norm(dst_pts - projected_pts, axis=2)
        mean_error = np.mean(errors)
        return mean_error

    def infer_current_gps(self, current_image, current_index):
        """Infer the current GPS location based on the highest correlated stored image."""
        current_features = detect_and_compute_superpoint(current_image)

        max_matches = 0
        best_homography = None
        best_index = -1
        min_reprojection_error = float('inf')

        for i in range(current_index):
            result = self._compute_homography(current_features, self.stored_features[i])
            if result is None:
                continue

            M, mask, matches = result
            good_matches = np.sum(mask)

            if good_matches > max_matches:
                reprojection_error = self.calculate_reprojection_error(
                    self.stored_features[i]['keypoints'][matches[:, 0]].cpu().numpy().reshape(-1, 1, 2),
                    current_features['keypoints'][matches[:, 1]].cpu().numpy().reshape(-1, 1, 2),
                    M
                )
                if reprojection_error < min_reprojection_error:
                    min_reprojection_error = reprojection_error
                    max_matches = good_matches
                    best_homography = M
                    best_index = i

        if best_homography is not None and best_index != -1:
            h, w = self.stored_images[best_index].shape[:2]
            center_pt = np.array([[w / 2, h / 2]], dtype='float32').reshape(-1, 1, 2)
            transformed_center = cv2.perspectiveTransform(center_pt, best_homography)
            delta_x, delta_y = transformed_center[0][0] - center_pt[0][0]

            gps_x, gps_y = self.stored_gps[best_index]
            current_gps_x = gps_x + delta_x * self.gps_per_pixel_x
            current_gps_y = gps_y + delta_y * self.gps_per_pixel_y

            return best_index, (current_gps_x, current_gps_y), min_reprojection_error
        return None, None, None

def parse_gps(file_path):
    """Parse GPS coordinates from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lat_str = lines[0].strip()
        lon_str = lines[1].strip()
        
        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_str)
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_str)

        lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
        lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)

        return lat, lon

def parse_dms(dms_str):
    """Parse degrees, minutes, seconds from a DMS string."""
    dms_str = dms_str.replace('°', ' ').replace('\'', ' ').replace('"', ' ')
    parts = dms_str.split()
    deg = int(parts[0])
    min = int(parts[1])
    sec = float(parts[2])
    dir = parts[3]
    return deg, min, sec, dir

def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0
    if dir in ['S', 'W']:
        decimal = -decimal
    return decimal

def calculate_optimal_gps_conversion(actual_gps_list, estimated_gps_list):
    """Calculate the optimal GPS to pixel conversion factors using optimization."""

    def error_function(factors):
        gps_per_pixel_x, gps_per_pixel_y = factors
        total_error = 0.0

        for actual, estimated in zip(actual_gps_list, estimated_gps_list):
            # Calculate the difference
            estimated_lon = actual[0] + (estimated[0] - actual[0]) * gps_per_pixel_x
            estimated_lat = actual[1] + (estimated[1] - actual[1]) * gps_per_pixel_y

            total_error += np.sqrt((estimated_lon - actual[0]) ** 2 + (estimated_lat - actual[1]) ** 2)

        return total_error

    # Initial guess for the conversion factors
    initial_guess = [0.01, 0.01]
    result = minimize(error_function, initial_guess, method='Nelder-Mead')

    return result.x  # Return the optimized factors

def main():
    gps_per_pixel_x = 0.403e-3  # Longitude
    gps_per_pixel_y = 0.403e-3  # Latitude
    navigator = UAVNavigator(gps_per_pixel_x, gps_per_pixel_y)

    # Directory containing the images and GPS files
    directory = './GoogleEarth/SET1'

    # Lists to store actual and estimated GPS coordinates
    actual_gps_list = []
    estimated_gps_list = []

    # Add images and GPS coordinates to the navigator
    for i in range(1, 7):
        image_path = os.path.join(directory, f'{i}.jpg')
        gps_path = os.path.join(directory, f'{i}.txt')
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gps_coordinates = parse_gps(gps_path)
        
        navigator.add_image(image, gps_coordinates)

    # Simulate backward flight (GPS lost)
    for i in range(5, 0, -1):
        current_image = navigator.stored_images[i]
        best_index, estimated_gps, reprojection_error = navigator.infer_current_gps(current_image, i)

        if best_index is not None and estimated_gps is not None:
            actual_gps = navigator.stored_gps[i]
            actual_gps_list.append(actual_gps)
            estimated_gps_list.append(estimated_gps)
            print(f"Image {i+1} best match with the following: Image {best_index+1}")
            print(f"Estimated GPS Location for image {i+1}: {estimated_gps}")
            print(f"Actual GPS Location for image {i+1}: {actual_gps}")
            print(f"Reprojection Error: {reprojection_error:.2f} pixels")
            print(f"Deviation-x (%) = {abs(estimated_gps[0] - actual_gps[0]) / abs(actual_gps[0]) * 100}")
            print(f"Deviation-y (%) = {abs(estimated_gps[1] - actual_gps[1]) / abs(actual_gps[1]) * 100}")
            print(f"Deviation-x (m) = {abs(estimated_gps[0] - actual_gps[0]) * 111139}")
            print(f"Deviation-y (m) = {abs(estimated_gps[1] - actual_gps[1]) * 111139}")
        else:
            print(f"Image {i+1}: Unable to estimate GPS location")

    # Calculate the optimal GPS to pixel conversion factors
    optimal_factors = calculate_optimal_gps_conversion(actual_gps_list, estimated_gps_list)
    print(f"Optimal GPS to Pixel Conversion Factors: Longitude = {optimal_factors[0]}, Latitude = {optimal_factors[1]}")

    # Plot the actual and estimated GPS coordinates for longitude
    actual_gps_x = [coord[0] for coord in actual_gps_list]
    actual_gps_y = [coord[1] for coord in actual_gps_list]
    estimated_gps_x = [coord[0] for coord in estimated_gps_list]
    estimated_gps_y = [coord[1] for coord in estimated_gps_list]

    plt.figure()
    plt.plot(range(1, len(actual_gps_x)+1), actual_gps_x, label='Actual Longitude')
    plt.plot(range(1, len(estimated_gps_x)+1), estimated_gps_x, label='Estimated Longitude')
    plt.xlabel('Image Index')
    plt.ylabel('Longitude')
    plt.legend()
    plt.title('Actual vs Estimated GPS Longitude')

    plt.figure()
    plt.plot(range(1, len(actual_gps_y)+1), actual_gps_y, label='Actual Latitude')
    plt.plot(range(1, len(estimated_gps_y)+1), estimated_gps_y, label='Estimated Latitude')
    plt.xlabel('Image Index')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('Actual vs Estimated GPS Latitude')
    plt.show()

if __name__ == "__main__":
    main()
