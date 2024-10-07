import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations
import torch  # For deep learning
from sklearn.linear_model import LinearRegression
from lightglue import LightGlue, SuperPoint  # For feature extraction
from lightglue.utils import rbd  # Utility function for removing batch dimension

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SuperPoint and LightGlue
extractor = SuperPoint(
    max_num_keypoints=2048,  # Maximum number of keypoints
    detection_threshold=0.0000015,  # Detection threshold. lower is more keypoints
    nms_radius=5  # Non-maximum suppression radius
).eval().to(device)  # Set the model to evaluation mode and move it to the device

matcher = LightGlue(  # Initialize LightGlue
    features='superpoint',  # Use SuperPoint features
    depth_confidence=0.95,
    width_confidence=0.99,
    filter_threshold=0.00155  # Custom filter threshold
).eval().to(device)

best_matches_matcher = LightGlue(  # Initialize LightGlue for best matches
    features='superpoint',  # Use SuperPoint features
    depth_confidence=0.95,
    width_confidence=0.99,
    filter_threshold=0.0155  # Custom filter threshold for best matches
).eval().to(device)

def normalize_image(image):
    """Normalize the image to [0,1], add batch and channel dimensions, and convert to torch tensor."""
    if len(image.shape) == 2:  # Ensure the image is grayscale
        image = image[None, None, :, :]  # Add batch and channel dimensions: (1, 1, H, W)
    else:
        raise ValueError("Input image must be grayscale.")
    
    return torch.tensor(image / 255.0, dtype=torch.float32).to(device)

def detect_and_compute_superpoint(image):
    """Detect and compute SuperPoint features."""
    normalized_image = normalize_image(image)
    feats = extractor.extract(normalized_image)
    return feats

def match_features_lightglue(featsA, featsB):
    """Match features using LightGlue."""
    matches = matcher({'image0': featsA, 'image1': featsB})
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]
    return featsA, featsB, matches['matches']

def best_match_lightglue(featsA, featsB):
    """Match features using LightGlue for best matches."""
    matches = best_matches_matcher({'image0': featsA, 'image1': featsB})
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]] #rbd is a utility function that removes the batch dimension. full dimension: (1, 1, H, W) -> (H, W). 
    return featsA, featsB, matches['matches']

class UAVNavigator:
    def __init__(self, gps_to_pixel_scale):
        self.gps_to_pixel_scale = gps_to_pixel_scale  # Pixels per meter
        self.stored_images = []
        self.stored_features = []
        self.stored_gps = []
        self.inferred_factor_x = 1
        self.inferred_factor_y = 1
        self.estimations_x = []
        self.estimations_y = []
        self.actuals_x = []
        self.actuals_y = []

    def clear_stored_data(self):
        """Clear stored images, features, and GPS data."""
        self.stored_images = []
        self.stored_features = []
        self.stored_gps = []
        self.estimations_x = []
        self.estimations_y = []
        self.actuals_x = []
        self.actuals_y = []

    def crop_image(self, image, kernel_to_test):
        """Crop the top and bottom 10% of the image."""
        height = image.shape[0]
        cropped_image = image[int(height * 0.1):int(height * 0.9), :]
        cropped_image = cv2.GaussianBlur(cropped_image, (kernel_to_test, kernel_to_test), 0)  # Denoise
        return cropped_image

    def add_image(self, image, gps_coordinates, kernel_to_test):
        """Add an image and its GPS coordinates to the stored list."""
        cropped_image = self.crop_image(image, kernel_to_test)
        
        if len(cropped_image.shape) == 3:  # Convert to grayscale if the image is not already
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        features = detect_and_compute_superpoint(cropped_image)
        # in algorithmic techniques, extractors extract keypoints and descriptors. Here, features are returned from the detect_and_compute_superpoint function. Features have many keys inside including: keypoints, scores, descriptors, etc. we can use the scores to do initial keypoint screening. 
        self.stored_images.append(cropped_image)
        self.stored_features.append(features)
        self.stored_gps.append(gps_coordinates)

    def compute_linear_regression_factors(self):
        """Compute the linear regression factors for both x and y based on the stored estimates and actual values."""
        
        # Prepare data for linear regression
        x_estimates = np.array(self.estimations_x).reshape(-1, 1)
        y_estimates = np.array(self.estimations_y).reshape(-1, 1)
            
        x_actual = np.array(self.actuals_x)
        y_actual = np.array(self.actuals_y)
        
        if len(x_estimates) > 0 and len(y_estimates) > 0:
            # Perform linear regression for x
            reg_x = LinearRegression().fit(x_estimates, x_actual)
            inferred_factor_x = reg_x.coef_[0]
            print(f"Linear regression inferred factor x: {inferred_factor_x}")

            # Perform linear regression for y
            reg_y = LinearRegression().fit(y_estimates, y_actual)
            inferred_factor_y = reg_y.coef_[0]
            print(f"Linear regression inferred factor y: {inferred_factor_y}")

            # Update the inferred factors
            self.inferred_factor_x = inferred_factor_x
            self.inferred_factor_y = inferred_factor_y
        else:
            print("Not enough data points to perform linear regression.")

    def compute_pixel_shifts_and_rotation(self, feats1, feats2, mode, lower_percentile=20, upper_percentile=80):
        """Compute pixel shifts and rotation using matched features."""
        if mode == 0:
            feats1, feats2, matches = best_match_lightglue(feats1, feats2)
        else:
            feats1, feats2, matches = match_features_lightglue(feats1, feats2)

        keypoints1 = feats1['keypoints'].cpu().numpy() # the .cpu() is used to move the tensor to the cpu. The .numpy() is used to convert the tensor to a numpy array
        keypoints2 = feats2['keypoints'].cpu().numpy()
        matches = matches.cpu().numpy()

        if len(matches) < 4:
            print("Warning: Less than 4 matches found.")
            return None, None, None

        src_pts = keypoints1[matches[:, 0]]
        dst_pts = keypoints2[matches[:, 1]]
        shifts = dst_pts - src_pts

        # Filter out outliers in the shifts (jointly for x and y)
        shifts = self.filter_outliers_joint(shifts, lower_percentile, upper_percentile)

        # Check if we have valid shifts after filtering
        if shifts is None or len(shifts) == 0:
            print("Warning: No valid pixel shifts after filtering. Skipping.")
            return None, None, None

        # Estimate the homography matrix using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
            return shifts, angle, len(matches)

        print("Warning: Homography could not be estimated.")
        return shifts, None, len(matches)

    def filter_outliers_joint(self, shifts, lower_percentile, upper_percentile, std_devs=2):
        """Filter out outliers in shifts jointly, ensuring corresponding x and y components stay aligned."""
        if len(shifts) == 0:
            return shifts

        # Calculate norms (distances) of shifts for outlier filtering
        norms = np.linalg.norm(shifts, axis=1)

        # Filter by percentiles
        lower_bound = np.percentile(norms, lower_percentile)
        upper_bound = np.percentile(norms, upper_percentile)
        mask = (norms >= lower_bound) & (norms <= upper_bound)
        filtered_shifts = shifts[mask]

        # Further filter by standard deviation
        if len(filtered_shifts) > 0:
            mean = np.mean(filtered_shifts, axis=0)
            std_dev = np.std(filtered_shifts, axis=0)
            if np.any(std_dev > 0):
                mask = np.all(np.abs(filtered_shifts - mean) <= std_devs * std_dev, axis=1)
                filtered_shifts = filtered_shifts[mask]

        return filtered_shifts

    def rotate_image(self, image, angle):
        """Rotate the image by a given angle around its center."""
        if angle is None:
            return image

        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    def translate_image(self, image, shift_x, shift_y):
        """Translate the image by a given x and y shift."""
        height, width = image.shape[:2]
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
        return translated_image

    def analyze_matches(self, lower_percentile, bool_infer_factor, num_images_analyse, image_offset=0):
        deviation_norms_x = []
        deviation_norms_y = []
        rotations_arr = []

        range_im = num_images_analyse if bool_infer_factor else len(self.stored_images)
        
        for i in reversed(range(1, range_im)):  # iterate through all images in reverse order
            best_index = -1
            max_good_matches = 0

            for j in range(i):  # iterate through all images before the current image
                shifts, angle, num_good_matches = self.compute_pixel_shifts_and_rotation(
                    self.stored_features[i], self.stored_features[j], 
                    mode=0
                )

                if shifts is not None and num_good_matches > max_good_matches:  # if the number of good matches is greater than the current max, update the max and the best index
                    max_good_matches = num_good_matches
                    best_index = j

            if best_index != -1:
                best_index = i - 1
                shifts, angle, num_good_matches = self.compute_pixel_shifts_and_rotation(
                    self.stored_features[i], self.stored_features[best_index], 
                    mode=1
                )
                cumul_ang = 0

                if shifts is not None:
                    # Apply rotation correction if angle is not None
                    if angle is not None:
                        rotated_image = self.stored_images[best_index]
                        for _ in range(1, 2, 1):  # Iterate to refine the rotation angle
                            rotated_image = self.rotate_image(rotated_image, angle)
                            cumul_ang += angle
                            rotated_features = detect_and_compute_superpoint(rotated_image)
                            shifts, angle, num_good_matches = self.compute_pixel_shifts_and_rotation(
                                self.stored_features[i], rotated_features, mode=1
                            )
                            # translate the image
                            if shifts is not None:
                                shift_x = np.mean(shifts[:, 0])
                                shift_y = np.mean(shifts[:, 1])
                                rotated_image = self.translate_image(rotated_image, shift_x, shift_y)
                            
                    rotations_arr.append(cumul_ang)  # Append rotation to the array

                    pixel_changes_x = shifts[:, 0]
                    pixel_changes_y = shifts[:, 1]

                    if len(pixel_changes_x) == 0 or len(pixel_changes_y) == 0:
                        print(f"Warning: No valid pixel changes after filtering for images {i+1} and {best_index+1}. Skipping.")
                        continue

                    actual_gps_diff = np.array(self.stored_gps[i]) - np.array(self.stored_gps[best_index])
                    actual_gps_diff_meters = actual_gps_diff * 111139  # Convert degrees to meters

                    actual_pixel_change_x = actual_gps_diff_meters[0] * self.gps_to_pixel_scale
                    actual_pixel_change_y = actual_gps_diff_meters[1] * self.gps_to_pixel_scale
                    
                    if bool_infer_factor and actual_pixel_change_x != 0 and actual_pixel_change_y != 0:
                        # Estimate linear regression factors with mse function
                        self.estimations_x.append(np.mean(pixel_changes_x))
                        self.estimations_y.append(np.mean(pixel_changes_y))
                        self.actuals_x.append(actual_pixel_change_x)
                        self.actuals_y.append(actual_pixel_change_y)

                    pixel_changes_y = pixel_changes_y * self.inferred_factor_y
                    pixel_changes_x = pixel_changes_x * self.inferred_factor_x

                    mean_pixel_changes_x = np.mean(pixel_changes_x)
                    mean_pixel_changes_y = np.mean(pixel_changes_y)

                    deviation_x_meters = mean_pixel_changes_x - actual_pixel_change_x
                    deviation_y_meters = mean_pixel_changes_y - actual_pixel_change_y
                    deviation_norms_x.append(np.abs(deviation_x_meters))
                    deviation_norms_y.append(np.abs(deviation_y_meters))
                    print(f"Deviations: {deviation_x_meters} meters (x), {deviation_y_meters} meters (y) for image {i+1}")

        return None, None, None, None, np.mean(deviation_norms_x), np.mean(deviation_norms_y), rotations_arr

def parse_gps(file_path):
    """Parse GPS coordinates from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lon_str = lines[1].strip()  # (x/lon is second in notepad)
        lat_str = lines[0].strip()  # lat is y (first line)
        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_str)
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_str)
        lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
        lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)
        return lon, -lat  # invert y-axis as it's defined as South First. return x first then y

def mse_function(inferred_factors, actual_pixel_changes_x, actual_pixel_changes_y, mean_pixel_changes_x, mean_pixel_changes_y):
    inferred_factor_x, inferred_factor_y = inferred_factors
    
    # Calculate the estimated pixel changes
    estimated_pixel_changes_x = mean_pixel_changes_x * inferred_factor_x
    estimated_pixel_changes_y = mean_pixel_changes_y * inferred_factor_y
    
    # Compute MSE
    mse_x = np.mean((actual_pixel_changes_x - estimated_pixel_changes_x) ** 2)
    mse_y = np.mean((actual_pixel_changes_y - estimated_pixel_changes_y) ** 2)
    
    return mse_x + mse_y

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

def main():
    gps_to_pixel_scale = 596 / 1092  # Pixels per meter
    navigator = UAVNavigator(gps_to_pixel_scale)
    directory = './GoogleEarth/SET1'
    num_images = 13
    lower_percentiles = [20]
    normalized_errors_percentiles = []
    kernels_to_test = [3]
    iteration_count = 0
    bool_infer_factor = True  # Enable factor inference during image addition

    for kernel in kernels_to_test:
        normalized_errors_percentiles = []
        iteration_count += 1
        for lower_percentile in lower_percentiles:
            navigator.clear_stored_data()  # Clear stored data before each kernel test

            # Step 1: Add images and infer factors
            for i in range(1, num_images + 1):
                image_path = os.path.join(directory, f'{i}.jpg')
                gps_path = os.path.join(directory, f'{i}.txt')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
                gps_coordinates = parse_gps(gps_path)
                navigator.add_image(image, gps_coordinates, kernel)

            # Run analysis to infer factors
            _, _, _, _, _, _, rotations = navigator.analyze_matches(lower_percentile, bool_infer_factor, num_images)
            navigator.compute_linear_regression_factors()
            print("INFERRED FACTORS:", navigator.inferred_factor_x, navigator.inferred_factor_y)

            # Step 2: Clear data and perform actual analysis
            navigator.clear_stored_data()  # Clear stored data before actual analysis
            
            # Add images again for actual analysis
            for i in range(1, num_images + 1):
                image_path = os.path.join(directory, f'{i}.jpg')
                gps_path = os.path.join(directory, f'{i}.txt')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
                gps_coordinates = parse_gps(gps_path)
                navigator.add_image(image, gps_coordinates, kernel)

            # Run actual analysis with inferred factors
            _, _, _, _, mean_x_dev, mean_y_dev, rotations = navigator.analyze_matches(lower_percentile, False, num_images)
            print(f'Array of rotations: {rotations}')
            print("Mean normalized error", np.linalg.norm([mean_x_dev, mean_y_dev]))
            normalized_error = np.linalg.norm([mean_x_dev, mean_y_dev])
            normalized_errors_percentiles.append(normalized_error)
            print(f'Iteration: {iteration_count}, Lower Percentile: {lower_percentile}')
            
    print(normalized_errors_percentiles)

if __name__ == "__main__":
    main()
