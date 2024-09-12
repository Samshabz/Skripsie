import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations
import matplotlib.pyplot as plt  # For plotting
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
import time
from Local_Matchers import set_matcher


def compare_histograms(img1, img2):
    # Convert both images to HSV
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Convert both images to HSV
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    # Compute histograms for both images
    hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])

    # Normalize the histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Compare histograms using correlation
    score1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    score2, _ = ssim(img1_gray, img2_gray, full=True)


    hash1 = imagehash.average_hash(img1_pil)
    hash2 = imagehash.average_hash(img2_pil)
    score3 = 1 - (hash1 - hash2) / len(hash1.hash)  # Normalized Hamming distance
    #print(f"All scores: {score1}, {score2}, {score3}")
    score = (score2+score3+score1) / 3
    # the 3 methods are score1: histogram correlation, score2: SSIM, score3: average hash's hamming distance

    
    return score

class UAVNavigator:
    def __init__(self, detector_choice):
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

        # Initialize the detector based on choice
        if detector_choice == 1:
            self.detector_name = "ORB"
            self.detector = cv2.ORB_create(nfeatures=20000)
            self.detector.setEdgeThreshold(10)  # Lower edge threshold (default is 31). range is 0-100. faster is lower
            self.detector.setFastThreshold(5)   # Lower FAST threshold (default is 20). range is 1-100. faster is lower

        elif detector_choice == 2:
            self.detector_name = "AKAZE"
            self.detector = cv2.AKAZE_create(threshold=0.0005)  # Lower threshold (default is 0.001). faster is lower


        else:
            raise ValueError("Invalid detector choice! Please choose 1 for ORB, 2 for AKAZE")

        search_params = dict(checks=150)
        if 0==1:
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = set_matcher("flann_matcher")#cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # both akaze and orb use binary descriptors - norm hamming
            # Supported options: "bf_matcher", "flann_matcher", "lsh_matcher", "ann_matcher", "graph_matcher"



         

    def find_best_match(self, image_index, grid_size=(5, 5), lower_percentile=20, upper_percentile=100-20):
        """Finds the best matching stored image for the given image index, using homography for rotation correction and grid-based ORB feature matching."""
        
        # Start timer to measure performance
        start_time = time.time()

        if len(self.stored_descriptors) == 0:
            raise ValueError("No descriptors available for matching.")
        
        # Function to divide image into grids
        def divide_into_grids(image, grid_size):
            """Divides the given image into grid_size (rows, cols) and returns the grid segments."""
            height, width = image.shape[:2]
            grid_height = height // grid_size[0]
            grid_width = width // grid_size[1]
            grids = []

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    grid = image[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]
                    grids.append(grid)
            return grids

        best_index = -1
        max_corr_score = -np.inf

        current_image = pull_image(image_index)
        current_grids = divide_into_grids(current_image, grid_size)

        for i in range(image_index):
            if i >= image_index:
                continue

            stored_image = pull_image(i)
            kp_current = self.stored_keypoints[image_index]
            descriptors_current = self.stored_descriptors[image_index]

            kp_stored = self.stored_keypoints[i]
            descriptors_stored = self.stored_descriptors[i]

            # Match descriptors between the current image and the stored image
            matches = self.matcher.find_matches(descriptors_current, descriptors_stored)  #knnMatch(descriptors_current, descriptors_stored, k=2)
            

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                        good_matches.append(m)

            if len(good_matches) > 10:
                # Extract matched points
                src_pts = np.float32([kp_current[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_stored[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography to estimate transformation
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Rotate the stored image using the homography matrix
                    h, w = stored_image.shape[:2]
                    rotated_image_stored = cv2.warpPerspective(stored_image, M, (w, h))

                    # Now divide both current and rotated stored image into grids
                    rotated_grids_stored = divide_into_grids(rotated_image_stored, grid_size)

                    total_good_matches = 0

                    # Perform grid-wise matching
                    for current_grid, stored_grid in zip(current_grids, rotated_grids_stored):
                        kp_current_grid, descriptors_current_grid = self.detector.detectAndCompute(current_grid, None)
                        kp_stored_grid, descriptors_stored_grid = self.detector.detectAndCompute(stored_grid, None)

                        if descriptors_current_grid is None or descriptors_stored_grid is None:
                            continue

                        # Match descriptors between grids using knnMatch
                        matches_grid = self.matcher.find_matches(descriptors_current_grid, descriptors_stored_grid)

                        good_matches_grid = []
                        for match_pair in matches_grid:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.999 * n.distance:  # Lowe's ratio test
                                    good_matches_grid.append(m)

                        # Sum the good matches for each grid
                        total_good_matches += len(good_matches_grid) if len(good_matches_grid) < 50 else 50

                    # Score based on total good matches across all grids
                    score = total_good_matches

                    if score > max_corr_score:
                        max_corr_score = score
                        best_index = i

        # End timer and calculate time taken
        total_time = time.time() - start_time


        print(f"Best match for image {image_index+1} is image {best_index+1} with a total of {max_corr_score} good matches.")
        print(f"Time taken: {total_time:.2f} seconds")
        
        return best_index




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
        
        # Convert to grayscale for detectors if not already
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
        keypoints, descriptors = self.detector.detectAndCompute(gray_image, None)
        
        # Check if descriptors are None
        if descriptors is None:
            print(f"Warning: No descriptors found for one image. Skipping.")
            return

        #self.stored_images.append(cropped_image)
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

    def compute_pixel_shifts_and_rotation(self, keypoints1, descriptors1, keypoints2, descriptors2, bool_calc_best_match, lower_percentile=20, upper_percentile=80):
        matches = self.matcher.find_matches(descriptors1, descriptors2)
        if bool_calc_best_match:
            # Find the best match for each keypoint
            match_ratio = 0.995
        else:
            match_ratio = 0.75
        # Apply ratio test as per Lowe's paper
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < match_ratio * n.distance:
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
        # lets print the angle we would have gotten if we used affine2Dpartial

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

        range_im = num_images_analyse if bool_infer_factor else 13#len(self.stored_images)
        num_correct_matches = 0
        for i in reversed(range(1, range_im)):  # iterate through all images in reverse order
            best_index = -1   
            
            best_index = self.find_best_match(i)
                    # Update correct_matches array
            if (best_index == (i - 1 if i!=10 else i-2)):
                num_correct_matches += 1
            print(f"Total correct matches at stage {i} is {num_correct_matches}")


            if best_index != -1:
                #print(f"Best match for image {i+1} is image {best_index+1} with undefined good matches.")
                #best_index = i - 1
                shifts, angle, num_good_matches = self.compute_pixel_shifts_and_rotation(
                    self.stored_keypoints[i], self.stored_descriptors[i], 
                    self.stored_keypoints[best_index], self.stored_descriptors[best_index], 0
                )
                
                cumul_ang = 0

                if shifts is not None:
                    # Apply rotation correction if angle is not None
                    if angle is not None:
                        rotated_image = pull_image(best_index) #self.stored_images[best_index]
                        for _ in range(1, 3, 1):  # Iterate to refine the rotation angle
                            rotated_image = self.rotate_image(rotated_image, angle)
                            cumul_ang += angle
                            rotated_keypoints, rotated_descriptors = self.detector.detectAndCompute(rotated_image, None)
                            shifts, angle, num_good_matches = self.compute_pixel_shifts_and_rotation(
                                self.stored_keypoints[i], self.stored_descriptors[i], 
                                rotated_keypoints, rotated_descriptors, 0
                            )
                            # translate the image
                            if shifts is not None:
                                shift_x = np.mean(shifts[:, 0])
                                shift_y = np.mean(shifts[:, 1])
                                #rotated_image = self.translate_image(rotated_image, shift_x, shift_y)
                            
                    rotations_arr.append(cumul_ang)  # Append rotation to the array

                    pixel_changes_x = shifts[:, 0]
                    pixel_changes_y = shifts[:, 1]

                    if len(pixel_changes_x) == 0 or len(pixel_changes_y) == 0:
                        print(f"Warning: No valid pixel changes after filtering for images {i+1} and {best_index+1}. Skipping.")
                        continue

                    actual_gps_diff = np.array(self.stored_gps[i]) - np.array(self.stored_gps[best_index])
                    actual_gps_diff_meters = actual_gps_diff * 111139  # Convert degrees to meters

                    actual_pixel_change_x = actual_gps_diff_meters[0] 
                    actual_pixel_change_y = actual_gps_diff_meters[1] 
                    
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


        return np.mean(deviation_norms_x), np.mean(deviation_norms_y), rotations_arr

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


def overlay_images(img1, img2, alpha=0.5):
    """
    Overlays two images and displays the result.

    Parameters:
    - image_path1: Path to the first image (background).
    - image_path2: Path to the second image (overlay).
    - alpha: Weight of the first image (0 to 1). The second image will have (1 - alpha) weight.
    """
    # Read the images


    if img1 is None or img2 is None:
        print("Error: One of the images could not be loaded.")
        return

    # Ensure both images have the same size
    if img1.shape != img2.shape:
        print("Error: Images do not have the same dimensions.")
        return

    # Blend the images
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

    # Display the result
    cv2.imshow('Blended Image', blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_outer_image(image, kernel_to_test):
    """Crop the top and bottom 10% of the image."""
    height = image.shape[0]
    cropped_image = image[int(height * 0.1):int(height * 0.9), :]
    cropped_image = cv2.GaussianBlur(cropped_image, (kernel_to_test, kernel_to_test), 0)  # Denoise
    return cropped_image

def pull_image(index):
    """Pull an image from from the directory with name index.jpg"""
    directory = './GoogleEarth/SET1'
    image_path = os.path.join(directory, f'{index+1}.jpg')
    
    # read in grey 
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
    # read in grey
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in
    cropped_image = crop_outer_image(image, 3)
        
        # Convert to grayscale for detectors if not already
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
    return gray_image

def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0
    if dir in ['S', 'W']:
        decimal = -decimal
    return decimal

def main():
    
    detector_choice = 2  # Set 1 for ORB, 2 for AKAZE, 3 for SURF *NOT FREE*
    navigator = UAVNavigator(detector_choice)

    directory = './GoogleEarth/SET1'
    num_images = 13
    inference_images = 7
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
            for i in range(1, inference_images + 1): # stops at inference_images = 6
                image_path = os.path.join(directory, f'{i}.jpg')
                gps_path = os.path.join(directory, f'{i}.txt')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
                gps_coordinates = parse_gps(gps_path)
                navigator.add_image(image, gps_coordinates, kernel)

            # Run analysis to infer factors
            _, _, rotations = navigator.analyze_matches(lower_percentile, bool_infer_factor, inference_images)
            navigator.compute_linear_regression_factors()
            print("INFERRED FACTORS:", navigator.inferred_factor_x, navigator.inferred_factor_y)
            # flush estimations prior to factor inference
            navigator.estimations_x = []
            navigator.estimations_y = []
            # Add images again for actual analysis. should not have to do this. its not meant to be changed in the prior run. xxx
            # here we will implement a function which says if stream.available, then if gps_available add image, else we will do the second step which is inferring the GPS and heading for that image. 
            for i in range(inference_images+1, num_images + 1):
                image_path = os.path.join(directory, f'{i}.jpg')
                gps_path = os.path.join(directory, f'{i}.txt')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
                gps_coordinates = parse_gps(gps_path)
                navigator.add_image(image, gps_coordinates, kernel)
            # Run actual analysis with inferred factors
            mean_x_dev, mean_y_dev, rotations = navigator.analyze_matches(lower_percentile, False, num_images)
            print(f'Array of rotations: {rotations}')
            print("Mean normalized error", np.linalg.norm([mean_x_dev, mean_y_dev]))
            normalized_error = np.linalg.norm([mean_x_dev, mean_y_dev])
            normalized_errors_percentiles.append(normalized_error)
            print(f'Iteration: {iteration_count}, Lower Percentile: {lower_percentile}')
            
    print(normalized_errors_percentiles)

if __name__ == "__main__":
    main() 
