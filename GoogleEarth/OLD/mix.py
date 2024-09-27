import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations

class UAVNavigator:
    def __init__(self, gps_to_pixel_scale):
        """Initialize the UAVNavigator with GPS to pixel scale."""
        self.gps_to_pixel_scale = gps_to_pixel_scale  # Pixels per meter
        self.stored_images = []
        self.stored_keypoints = []
        self.stored_descriptors = []
        self.stored_gps = []

        # Initialize the SIFT detector
        self.sift = cv2.SIFT_create()

        # Set up FLANN-based matcher parameters for SIFT (floating-point features)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=100)  # Increase the number of checks
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def clear_stored_data(self):
        """Clear stored images, keypoints, descriptors, and GPS data."""
        self.stored_images.clear()
        self.stored_keypoints.clear()
        self.stored_descriptors.clear()
        self.stored_gps.clear()

    def crop_image(self, image, kernel_size):
        """Crop the top and bottom 10% of the image and apply Gaussian blur."""
        height = image.shape[0]
        cropped_image = image[int(height * 0.1):int(height * 0.9), :]
        return cv2.GaussianBlur(cropped_image, (kernel_size, kernel_size), 0)

    def add_image(self, image, gps_coordinates, kernel_size):
        """Add an image and its GPS coordinates to the stored list."""
        cropped_image = self.crop_image(image, kernel_size)
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

        if descriptors is None:
            print("Warning: No descriptors found for one image. Skipping.")
            return

        self.stored_images.append(cropped_image)
        self.stored_keypoints.append(keypoints)
        self.stored_descriptors.append(descriptors)
        self.stored_gps.append(gps_coordinates)

    def find_rotations(self, keypoints1, descriptors1, keypoints2, descriptors2, severity=5):
        """Estimate the rotation angle using the homography matrix and weight by match confidence."""
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 4:
            print("Warning: Not enough good matches to estimate rotation. Skipping.")
            return None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
        if M is not None:
            angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)

            # Calculate confidence for each match based on distance
            distances = np.array([m.distance for m in good_matches])
            max_distance = np.max(distances)
            confidences = 1 - (distances / max_distance)  # Higher scores for better matches

            # Calculate weighted rotation change
            if severity != 0:
                weighted_angles = angle * (confidences ** severity)
                weighted_rotation_change = np.sum(weighted_angles) / np.sum(confidences ** severity)
            else:
                weighted_rotation_change = angle


            return weighted_rotation_change

        print("Warning: Homography could not be estimated. Skipping.")
        return None


    def rotate_image(self, image, angle):
        """Rotate the image by a given angle around its center."""
        if angle is None:
            return image

        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    def compute_pixel_shifts(self, keypoints1, descriptors1, keypoints2, descriptors2):
        """Compute pixel shifts between two images."""
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        correlation_matches = [m for m, n in matches if m.distance < 0.95 * n.distance]

        if len(good_matches) < 4:
            return None, None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        shifts = dst_pts - src_pts
        return shifts, len(correlation_matches)

    def filter_outliers(self, data, lower_percentile, upper_percentile, std_devs=2):
        """Filter data to include only values between specified percentiles,
        then remove points outside a specified number of standard deviations."""
        if len(data) == 0:
            return data

        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

        if len(filtered_data) == 0:
            return filtered_data

        mean = np.mean(filtered_data)
        std_dev = np.std(filtered_data)
        if std_dev == 0:
            return filtered_data

        return filtered_data[(filtered_data >= mean - std_devs * std_dev) & (filtered_data <= mean + std_devs * std_dev)]

    def analyze_matches(self, lower_percentile):
        """Analyze matches and calculate deviations in estimated GPS coordinates."""
        ratios_x, ratios_y = [], []
        deviation_norms_x, deviation_norms_y = [], []
        rotations_arr = []
        final_rotation = 0

        for i in reversed(range(1, len(self.stored_images))):
            max_good_matches = 0
            best_index = -1

            for j in range(i):
                angle = self.find_rotations(
                    self.stored_keypoints[i], self.stored_descriptors[i],
                    self.stored_keypoints[j], self.stored_descriptors[j]
                )

                if angle is None:
                    continue

                rotated_image = self.rotate_image(self.stored_images[j], angle)
                gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
                keypoints_rot, descriptors_rot = self.sift.detectAndCompute(gray_image, None)

                shifts, num_corr_matches = self.compute_pixel_shifts(
                    self.stored_keypoints[i], self.stored_descriptors[i],
                    keypoints_rot, descriptors_rot
                )
                print(f"Number of good matches between image {i+1} and rotated image {j+1}: {num_corr_matches}")
                if shifts is not None and num_corr_matches > max_good_matches:
                    max_good_matches = num_corr_matches
                    best_index = j
                    final_rotation = angle
                final_rotation = angle
            if best_index != -1:
                best_index = i - 1
                angle = final_rotation
                # -0.5 from angle then round to nearest degree
                angle = round(angle, 1) # the accuracy of the angle is limited

                rotated_image = self.rotate_image(self.stored_images[best_index], angle)
                gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
                keypoints_rot, descriptors_rot = self.sift.detectAndCompute(gray_image, None)

                shifts, num_corr_matches = self.compute_pixel_shifts(
                    self.stored_keypoints[i], self.stored_descriptors[i],
                    keypoints_rot, descriptors_rot
                )
                rotations_arr.append(angle)
                print("Rotations estimate: ", angle)

                if shifts is not None:
                    pixel_changes_x = shifts[:, 0].round().astype(int)
                    pixel_changes_y = shifts[:, 1].round().astype(int)

                    pixel_changes_x = self.filter_outliers(pixel_changes_x, lower_percentile, 100 - lower_percentile)
                    pixel_changes_y = self.filter_outliers(pixel_changes_y, lower_percentile, 100 - lower_percentile)

                    if len(pixel_changes_x) == 0 or len(pixel_changes_y) == 0:
                        print(f"Warning: No valid pixel changes after filtering for images {i+1} and {best_index+1}. Skipping.")
                        continue

                    temp = pixel_changes_x
                    pixel_changes_x = -pixel_changes_y * np.sqrt(3) * 0.9774624872432882  # Rotate by 60 degrees
                    pixel_changes_y = temp * 2 * 1.0212814396689858  # Stretch by a factor of 2

                    actual_gps_diff = np.array(self.stored_gps[i]) - np.array(self.stored_gps[best_index])
                    actual_gps_diff_meters = actual_gps_diff * 111139  # Convert degrees to meters

                    actual_pixel_change_x = (actual_gps_diff_meters[0] * self.gps_to_pixel_scale).round().astype(int)
                    actual_pixel_change_y = (actual_gps_diff_meters[1] * self.gps_to_pixel_scale).round().astype(int)

                    mean_pixel_changes_x = np.mean(pixel_changes_x)
                    mean_pixel_changes_y = np.mean(pixel_changes_y)

                    perfect_ratio_x = actual_pixel_change_x / mean_pixel_changes_x
                    perfect_ratio_y = actual_pixel_change_y / mean_pixel_changes_y

                    ratios_x.append(perfect_ratio_x)
                    ratios_y.append(perfect_ratio_y)

                    estimated_gps_diff_meters_x = mean_pixel_changes_x / self.gps_to_pixel_scale
                    estimated_gps_diff_meters_y = mean_pixel_changes_y / self.gps_to_pixel_scale
                    estimated_gps_diff_x = estimated_gps_diff_meters_x / 111139
                    estimated_gps_diff_y = estimated_gps_diff_meters_y / 111139

                    estimated_gps_x = self.stored_gps[best_index][0] + estimated_gps_diff_x
                    estimated_gps_y = self.stored_gps[best_index][1] + estimated_gps_diff_y
                    estimated_gps = (estimated_gps_x, estimated_gps_y)

                    deviation_x_meters = estimated_gps_diff_meters_x - actual_gps_diff_meters[0]
                    deviation_y_meters = estimated_gps_diff_meters_y - actual_gps_diff_meters[1]
                    deviation_norms_x.append(np.abs(deviation_x_meters))
                    deviation_norms_y.append(np.abs(deviation_y_meters))

                    print(f"Deviation X (meters): {deviation_x_meters} in GPS estimation of image {i+1} with image {best_index+1}")
                    print(f"Deviation Y (meters): {deviation_y_meters} in GPS estimation of image {i+1} with image {best_index+1}")

        return np.std(ratios_x), np.std(ratios_y), np.mean(ratios_x), np.mean(ratios_y), np.mean(deviation_norms_x), np.mean(deviation_norms_y), rotations_arr

def parse_gps(file_path):
    """Parse GPS coordinates from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lat_str, lon_str = lines[0].strip(), lines[1].strip()
        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_str)
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_str)
        lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
        lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)
        return lat, lon

def parse_dms(dms_str):
    """Parse degrees, minutes, seconds from a DMS string."""
    dms_str = dms_str.replace('°', ' ').replace('\'', ' ').replace('"', ' ')
    parts = dms_str.split()
    deg, min, sec, dir = int(parts[0]), int(parts[1]), float(parts[2]), parts[3]
    return deg, min, sec, dir

def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0
    return -decimal if dir in ['S', 'W'] else decimal

def main():
    gps_to_pixel_scale = 596 / 1092  # Pixels per meter
    navigator = UAVNavigator(gps_to_pixel_scale)
    directory = './GoogleEarth/SET1'
    num_images = 12
    lower_percentiles = [47]
    kernels_to_test = [3]
    iteration_count = 0

    for kernel in kernels_to_test:
        iteration_count += 1
        for lower_percentile in lower_percentiles:
            navigator.clear_stored_data()  # Clear stored data before each kernel test
            for i in range(1, num_images + 1):
                image_path = os.path.join(directory, f'{i}.jpg')
                gps_path = os.path.join(directory, f'{i}.txt')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
                gps_coordinates = parse_gps(gps_path)
                navigator.add_image(image, gps_coordinates, kernel)

            std_x, std_y, mean_x, mean_y, mean_x_dev, mean_y_dev, rotations = navigator.analyze_matches(lower_percentile)
            print("Rotations: ", rotations)

            normalized_error = np.linalg.norm([mean_x_dev, mean_y_dev])
            print(f'Iteration: {iteration_count}, Lower Percentile: {lower_percentile}')
            print(f'Normalized Error: {normalized_error}')

    # Plot normalized errors based on different percentiles
    # plt.plot(lower_percentiles, normalized_errors_percentiles, label=f'Kernel {kernel}')
    # plt.xlabel('Lower Percentile')
    # plt.ylabel('Normalized Error')
    # plt.title('Normalized Error vs Lower Percentile')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
