import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations
import matplotlib.pyplot as plt  # For plotting

class UAVNavigator:
    def __init__(self, gps_to_pixel_scale):
        self.gps_to_pixel_scale = gps_to_pixel_scale  # Pixels per meter
        self.stored_images = []
        self.stored_keypoints = []
        self.stored_descriptors = []
        self.stored_gps = []

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

    def crop_image(self, image, kernel_to_test):
        """Crop the top and bottom 10% of the image."""
        height = image.shape[0]
        cropped_image = image[int(height * 0.1):int(height * 0.9), :]
        cropped_image = cv2.GaussianBlur(cropped_image, (kernel_to_test, kernel_to_test), 0)  # Denoise
        return cropped_image

    def add_image(self, image, gps_coordinates, kernel_to_test):
        """Add an image and its GPS coordinates to the stored list."""
        cropped_image = self.crop_image(image, kernel_to_test)

        # Convert to grayscale for ORB
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        
        # Check if descriptors are None
        if descriptors is None:
            print(f"Warning: No descriptors found for one image. Skipping.")
            return

        self.stored_images.append(cropped_image)
        self.stored_keypoints.append(keypoints)
        self.stored_descriptors.append(descriptors)
        self.stored_gps.append(gps_coordinates)

    def compute_pixel_shifts(self, keypoints1, descriptors1, keypoints2, descriptors2):
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test as per Lowe's paper
        good_matches = []
        correlation_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                if m.distance < 0.9995 * n.distance:
                    correlation_matches.append(m)
        if len(good_matches) < 4:
            return None, None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])  # tuple stored y first
        shifts = dst_pts - src_pts
        return shifts, len(correlation_matches)

    def filter_outliers(self, data, lower_percentile, upper_percentile, std_devs=2):
        """Filter data to include only values between the lower and upper percentiles,
        then remove points outside a specified number of standard deviations."""
        if len(data) == 0:
            return data

        # Filter by percentiles
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

        if len(filtered_data) == 0:
            return filtered_data

        # Further filter by standard deviation
        mean = np.mean(filtered_data)
        std_dev = np.std(filtered_data)
        if std_dev == 0:
            return filtered_data

        return filtered_data[(filtered_data >= mean - std_devs * std_dev) & (filtered_data <= mean + std_devs * std_dev)]

    def analyze_matches(self, lower_percentile):
        ratios_x = []
        ratios_y = []
        deviation_norms_x = []
        deviation_norms_y = []
        for i in reversed(range(1, len(self.stored_images))):  # iterate through all images in reverse order. i is the current image index
            current_image = self.stored_images[i]
            best_index = -1
            max_good_matches = 0

            for j in range(i):  # iterate through all images before the current image
                
                shifts, numcorrmatchess = self.compute_pixel_shifts(self.stored_keypoints[i], self.stored_descriptors[i], self.stored_keypoints[j], self.stored_descriptors[j])  # compute pixel shifts between the current image and the previous image
                print(f"Number of good matches between image {i+1} and image {j+1}: {len(shifts)}")
                if shifts is not None and numcorrmatchess > max_good_matches:  # if the number of good matches is greater than the current max, update the max and the best index
                    max_good_matches = len(shifts)
                    best_index = j

            if best_index != -1:
                best_index = i - 1  # TO DO: fix this. The method of most matches is not working. Must use correlation method. Namely, image 9 is matching with 7//6 not 8. 
                shifts, numcorrmatchess = self.compute_pixel_shifts(self.stored_keypoints[i], self.stored_descriptors[i], self.stored_keypoints[best_index], self.stored_descriptors[best_index])
                if shifts is not None:
                    # Calculate pixel changes for x and y directions separately
                    pixel_changes_x = shifts[:, 0].round().astype(int)
                    pixel_changes_y = shifts[:, 1].round().astype(int)

                    # Filter out outliers using both percentiles and standard deviation
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

                    # Convert GPS differences to pixel changes for x and y
                    actual_pixel_change_x = (actual_gps_diff_meters[0] * self.gps_to_pixel_scale).round().astype(int)
                    actual_pixel_change_y = (actual_gps_diff_meters[1] * self.gps_to_pixel_scale).round().astype(int)


                    mean_pixel_changes_x = np.mean(pixel_changes_x)
                    mean_pixel_changes_y = np.mean(pixel_changes_y)

                    if mean_pixel_changes_x == 0 or mean_pixel_changes_y == 0:
                        print(f"Warning: Mean pixel changes are zero for images {i+1} and {best_index+1}. Skipping.")
                        continue

                    perfect_ratio_x = actual_pixel_change_x / mean_pixel_changes_x
                    perfect_ratio_y = actual_pixel_change_y / mean_pixel_changes_y
                    
                    ratios_x.append(perfect_ratio_x)
                    ratios_y.append(perfect_ratio_y)

                    # Calculate the estimated GPS difference
                    estimated_gps_diff_meters_x = mean_pixel_changes_x / self.gps_to_pixel_scale
                    estimated_gps_diff_meters_y = mean_pixel_changes_y / self.gps_to_pixel_scale
                    estimated_gps_diff_x = estimated_gps_diff_meters_x / 111139
                    estimated_gps_diff_y = estimated_gps_diff_meters_y / 111139

                    # Calculate estimated GPS coordinates
                    estimated_gps_x = self.stored_gps[best_index][0] + estimated_gps_diff_x
                    estimated_gps_y = self.stored_gps[best_index][1] + estimated_gps_diff_y
                    estimated_gps = (estimated_gps_x, estimated_gps_y)

                    # Calculate deviation in meters
                    deviation_x_meters = estimated_gps_diff_meters_x - actual_gps_diff_meters[0]
                    deviation_y_meters = estimated_gps_diff_meters_y - actual_gps_diff_meters[1]
                    deviation_norms_x.append(np.abs(deviation_x_meters))
                    deviation_norms_y.append(np.abs(deviation_y_meters))

                    #print(f"Actual GPS: {self.stored_gps[i]}")
                    #print(f"Estimated GPS: {estimated_gps}")
                    print(f"Deviation X (meters): {deviation_x_meters} in GPS estimation of image {i+1} with image {best_index+1}")
                    print(f"Deviation Y (meters): {deviation_y_meters} in GPS estimation of image {i+1} with image {best_index+1}")
                    
        return np.std(ratios_x), np.std(ratios_y), np.mean(ratios_x), np.mean(ratios_y), np.mean(deviation_norms_x), np.mean(deviation_norms_y)

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

def main():
    gps_to_pixel_scale = 596 / 1092  # Pixels per meter
    navigator = UAVNavigator(gps_to_pixel_scale)
    directory = './GoogleEarth/SET1'
    num_images = 11
    lower_percentiles = [47]
    normalized_errors_percentiles = []
    kernels_to_test = [3]
    iteration_count = 0
    for kernel in kernels_to_test:
        normalized_errors_percentiles = []
        iteration_count += 1
        for lower_percentile in lower_percentiles:
            navigator.clear_stored_data()  # Clear stored data before each kernel test
            for i in range(1, num_images + 1):  # Assuming there are 10 images
                image_path = os.path.join(directory, f'{i}.jpg')
                gps_path = os.path.join(directory, f'{i}.txt')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
                gps_coordinates = parse_gps(gps_path)
                navigator.add_image(image, gps_coordinates, kernel)

            std_x, std_y, mean_x, mean_y, mean_x_dev, mean_y_dev = navigator.analyze_matches(lower_percentile)
            # print(f'Kernel: {kernel}, Lower Percentile: {lower_percentile}')
            # print(f'Standard deviation of estimation X: {std_x}')
            # print(f'Standard deviation of Estimation Y: {std_y}')
            # print(f'Estimator stability factor X: {mean_x}')
            # print(f'Estimator stability factor Y: {mean_y}')
            # print(f'Mean error in X (meters): {mean_x_dev}')
            # print(f'Mean error in Y (meters): {mean_y_dev}\n')
            normalized_error = np.linalg.norm([mean_x_dev, mean_y_dev])
            normalized_errors_percentiles.append(normalized_error)
            print(f'Iteration: {iteration_count}, Lower Percentile: {lower_percentile}')
            
    # plot normalized errors based on different percentiles 
    print(normalized_errors_percentiles)
    # plt.plot(lower_percentiles, normalized_errors_percentiles, label=f'Kernel {kernel}')
    # plt.xlabel('Lower Percentile')
    # plt.ylabel('Normalized Error')
    # plt.title('Normalized Error vs Lower Percentile')
    # plt.legend()
    # plt.show()

    
if __name__ == "__main__":
    main()
