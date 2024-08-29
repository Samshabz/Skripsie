import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations
import matplotlib.pyplot as plt  # For plotting
from sklearn.linear_model import LinearRegression

class UAVNavigator:
    def __init__(self, gps_to_pixel_scale):
        self.gps_to_pixel_scale = gps_to_pixel_scale  # Pixels per meter
        self.stored_images = []
        self.stored_keypoints = []
        self.stored_descriptors = []
        self.stored_gps = []
        self.inferred_factor_x = 1
        self.inferred_factor_y = 1
        self.estimations_x = []
        self.estimations_y = []
        self.actuals_x = []
        self.actuals_y = []

        # Initialize the ORB detector
        self.orb = cv2.ORB_create(nfeatures=20000)

        # Set up FLANN-based matcher parameters for ORB (binary features)
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=12,  # Increase table number
                            key_size=20,  # Increase key size
                            multi_probe_level=2)  # Increase multi-probe level
        search_params = dict(checks=100)  # Increase the number of checks
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def clear_stored_data(self):
        """Clear stored images, keypoints, descriptors, and GPS data."""
        self.stored_images = []
        self.stored_keypoints = []
        self.stored_descriptors = []
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
        
        # Convert to grayscale for ORB if not already
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        
        # Check if descriptors are None
        if descriptors is None:
            print(f"Warning: No descriptors found for one image. Skipping.")
            return

        self.stored_images.append(cropped_image)
        self.stored_keypoints.append(keypoints)
        self.stored_descriptors.append(descriptors)
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

    def compute_pixel_shifts_and_rotation(self, keypoints1, descriptors1, keypoints2, descriptors2, lower_percentile=20, upper_percentile=80):
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test as per Lowe's paper
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            print("Warning: Less than 4 matches found.")
            return None, None, None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
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
            return shifts, angle, len(good_matches)

        print("Warning: Homography could not be estimated.")
        return shifts, None, len(good_matches)

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
                    self.stored_keypoints[i], self.stored_descriptors[i], 
                    self.stored_keypoints[j], self.stored_descriptors[j]
                )

                if shifts is not None and num_good_matches > max_good_matches:  # if the number of good matches is greater than the current max, update the max and the best index
                    max_good_matches = num_good_matches
                    best_index = j

            if best_index != -1:
                best_index = i - 1
                shifts, angle, num_good_matches = self.compute_pixel_shifts_and_rotation(
                    self.stored_keypoints[i], self.stored_descriptors[i], 
                    self.stored_keypoints[best_index], self.stored_descriptors[best_index]
                )
                cumul_ang = 0

                if shifts is not None:
                    # Apply rotation correction if angle is not None
                    if angle is not None:
                        rotated_image = self.stored_images[best_index]
                        for _ in range(1, 2, 1):  # Iterate to refine the rotation angle
                            rotated_image = self.rotate_image(rotated_image, angle)
                            cumul_ang += angle
                            rotated_keypoints, rotated_descriptors = self.orb.detectAndCompute(rotated_image, None)
                            shifts, angle, num_good_matches = self.compute_pixel_shifts_and_rotation(
                                self.stored_keypoints[i], self.stored_descriptors[i], 
                                rotated_keypoints, rotated_descriptors
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

        # if bool_infer_factor:
        #     self.clear_stored_data()  # Clear data if the function was used for inference
        
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
