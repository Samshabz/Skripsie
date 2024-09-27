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
from Feature_Extractors import set_feature_extractor
import torch
from shiftdetectors import estimate_shift_from_points

class UAVNavigator:
    def __init__(self, global_detector_choice, local_detector_choice, global_matcher_choice, local_matcher_choice, global_matching_technique):
        self.stored_image_count = 0
        self.stored_global_keypoints = []
        self.stored_global_descriptors = []
        self.stored_local_keypoints = []
        self.stored_local_descriptors = []
        self.stored_gps = []
        self.inferred_factor_x = 1
        self.inferred_factor_y = 1
        self.estimated_gps_y = []
        self.actuals_x = []
        self.actuals_y = []
        self.global_detector_name = ""
        self.local_detector_name = ""
        self.global_matcher = None
        self.local_matcher = None
        self.global_matcher_choice = 0
        self.neural_net_on = False


        # angles 
        self.stored_headings = []
        self.estimated_headings = []
        self.estimated_heading_deviations = []
        

        # estimated GPS
        self.estimated_gps = []
        self.estimated_gps_x_inference = []
        self.estimated_gps_y_inference = []
        self.norm_GPS_error = []





        # timing 
        self.time_best_match_function = 0
        self.runtime_rotational_estimator = []

        if global_detector_choice == 1:
            self.global_detector_name = "ORB"
            self.global_detector = set_feature_extractor(global_detector_choice)

        elif global_detector_choice == 2: 
            self.global_detector_name = "AKAZE"
            self.global_detector = set_feature_extractor(global_detector_choice)

        if local_detector_choice == 1:
            self.local_detector_name = "ORB"
            self.local_detector = set_feature_extractor(local_detector_choice)

        elif local_detector_choice == 2: 
            self.local_detector_name = "AKAZE"
            self.local_detector = set_feature_extractor(local_detector_choice)
        

        # Initialize the detector based on choice
        else:
            raise ValueError("Invalid detector choice! Please choose 1 for ORB, 2 for AKAZE")


        globchoice = "bf_matcher" if global_matcher_choice == 0 else "flann_matcher" if global_matcher_choice == 1 else "graph_matcher" if global_matcher_choice == 2 else "histogram_matcher" if global_matcher_choice == 3 else "SSIM_matcher"
        self.global_matcher_choice = global_matcher_choice
        locchoice = "bf_matcher" if local_matcher_choice == 0 else "flann_matcher" if local_matcher_choice == 1 else "graph_matcher"
        self.local_matcher_choice = local_matcher_choice


        self.global_matcher = set_matcher(globchoice, self.neural_net_on) # 0 for BFMatcher, 1 for FlannMatcher, 2 for graph matcher. 

        self.local_matcher = set_matcher(locchoice, self.neural_net_on) 

        self.global_matching_technique = global_matching_technique  
    
    def get_rotations(self, src_pts, dst_pts, method_to_use):
            homography_threshold = 25 if self.global_detector_name == "ORB" else 0.5
            new_params_ransac = 25 if self.global_detector_name == "AKAZE" else 0.5

            # choose next 5, 50
            homography_threshold = 0.5 # 5 50 
            M = None
            mask = None
            if method_to_use == 0:
                M, mask = cv2.estimateAffine2D(src_pts, dst_pts , method=cv2.RANSAC, ransacReprojThreshold=new_params_ransac)
            elif method_to_use == 1:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_threshold)

            return M, np.sum(mask)



    def find_best_match_multimethod(self, image_index, image_space, grid_size=(5, 5), window_size=(50, 50)):
        # print("Finding best match using multiple methods.")
        def cross_correlate(image1, image2):
            """Performs cross-correlation between two full images."""
            result = cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)
            _, max_corr, _, _ = cv2.minMaxLoc(result)
            return max_corr
        """Finds the best matching stored image for the given image index using cross-correlation with rotational normalization."""

        # Start timer to measure performance
        if len(self.stored_global_descriptors) == 0:
            raise ValueError("No descriptors available for matching.")

        best_index = -1
        max_corr_score = -np.inf  # Initial maximum correlation score

        current_image = pull_image(image_index)  # Load the current image
        kp_current = self.stored_global_keypoints[image_index]
        descriptors_current = self.stored_global_descriptors[image_index]

        time_A = time.time()
        score = 0

        linked_angles = []
        for i in image_space:
            if i == image_index:  # Can change this to if i >= image_index to only infer from lower indices
                continue

            stored_image = pull_image(i)
            kp_stored = self.stored_global_keypoints[i]
            descriptors_stored = self.stored_global_descriptors[i]

            # Match descriptors between the current image and the stored image
            global_detector_choice = 1 if self.global_detector_name == "ORB" else 2
            matches = self.global_matcher.find_matches(descriptors_current, descriptors_stored, kp_current, kp_stored, global_detector_choice, global_matcher_true=1)

            good_matches = []
            for match_pair in matches:
                if self.global_matcher_choice == 2:  # Graph matcher returns singles
                    good_matches.append(match_pair)
                else:
                    # BFMatcher and FlannMatcher return pairs
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.8 * n.distance:  # Lowe's ratio test
                            good_matches.append(m)


            # Rotate the image if there are sufficient good matches
            if len(good_matches) > 10:
                src_pts = np.float32([kp_current[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_stored[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


                # Find homography to estimate transformation
                angleB = 0
                method_to_use = 0 # 0 is affine, 1 is homography
                M, amt_inliers = self.get_rotations(src_pts, dst_pts, method_to_use) 
                # method_to_use = 5
                # # use fourier melin function with both images estimate_rotation_fourier_mellin
                # if method_to_use == 5:
                #     # overlay_images(current_image, stored_image)
                #     angleB = phase_correlation_rotation(current_image, stored_image)  

                
                if amt_inliers < 40:
                    continue

                # if M is not None:
                if 1==1:
                    # Rotate the stored image using the homography matrix
                    h, w = stored_image.shape[:2]
                    input_image = stored_image
                    if method_to_use == 0:
                        rotated_image_stored = cv2.warpAffine(input_image, M[:2, :], (input_image.shape[1], input_image.shape[0]))
                    elif method_to_use == 1:
                        rotated_image_stored = cv2.warpPerspective(input_image, M, (input_image.shape[1], input_image.shape[0]))

                    elif method_to_use == 5:
                        src = input_image
                        rotated_image_stored = cv2.warpAffine(src, cv2.getRotationMatrix2D((src.shape[1]//2, src.shape[0]//2), angleB, 1), (src.shape[1], src.shape[0]))



                    # Perform cross-correlation directly on the full image
                    if self.global_matching_technique == 3:
                        score = cross_correlate(current_image, rotated_image_stored)


                    elif self.global_matching_technique == 4:
                        score = self.compare_histograms(current_image, rotated_image_stored)
                        

                    elif self.global_matching_technique == 5:
                        score = self.compare_ssim(current_image, rotated_image_stored)
                        


                    elif self.global_matching_technique == 6:
                        score = self.compare_hash(current_image, rotated_image_stored)
                        
                    if score > max_corr_score:
                        max_corr_score = score
                        best_index = i
                        # temp_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
                        linked_angles.append(angleB)
                        print(f"Angle from image {image_index+1} to {best_index+1} is {angleB} degrees.")
                        # this is the angle between image_index, and best_index.
                        # to get the new heading we need to say: temp_angle + stored_headings[best_index] = new heading.





        # pull the last value from linked_angles array
        if len(linked_angles) > 0:
            heading_change = linked_angles[-1]
            print(f"Heading from image {image_index+1} to {best_index+1} is {heading_change} degrees vs actual {-self.stored_headings[image_index]+self.stored_headings[best_index]} degrees.")
            self.compare_headings(image_index, best_index, heading_change)

        time_B = time.time() - time_A

        self.time_best_match_function += time_B
        # print(f"Best match for image {image_index+1} is image {best_index+1}")
        if best_index == -1:
            print("No best match found.")
            # break the code:
            return None
        return best_index


    # Additional comparison methods needed for the function
    def compare_histograms(self, img1, img2):
        """Compares two images using histogram correlation."""
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) if len(img1.shape) == 3 else cv2.cvtColor(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV) if len(img2.shape) == 3 else cv2.cvtColor(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score-1

    def compare_ssim(self, img1, img2):
        """Compares two images using SSIM."""
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        score, _ = ssim(img1_gray, img2_gray, full=True)
        return score

    def compare_hash(self, img1, img2):
        """Compares two images using average hash."""
        img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        hash1 = imagehash.average_hash(img1_pil)
        hash2 = imagehash.average_hash(img2_pil)
        score = 1 - (hash1 - hash2) / len(hash1.hash)  # Normalized Hamming distance
        return score
    
    def compare_headings(self, image_index, best_index, estimated_heading_change):
        # Normalize angles to be within 0 to 360 or -180 to 180 range as needed
        def normalize_angle(angle):
            angle = angle % 360  # Normalize to [0, 360)
            if angle > 180:  # Normalize to [-180, 180)
                angle -= 360
            return angle

        # Calculate the new heading, then normalize it
        estimated_heading_change = normalize_angle(estimated_heading_change)
        self.stored_headings[best_index] = normalize_angle(self.stored_headings[best_index]) # this we have as we are using it to infer the rotation. 
        self.stored_headings[image_index] = normalize_angle(self.stored_headings[image_index]) # this is only used for error comparison. 


        estimated_new_heading = (self.stored_headings[best_index] + estimated_heading_change) 
        # print(f"stored heading: {self.stored_headings[best_index]}")
        # print(f"estimated heading change: {estimated_heading_change}")
        estimated_new_heading = normalize_angle(estimated_new_heading)

        # Calculate the deviation and normalize it as well
        deviation_heading = self.stored_headings[image_index] - estimated_new_heading
        # print(f"deviation heading: {deviation_heading}")
        deviation_heading = normalize_angle(deviation_heading)
        print(f"ANGLE OFFSET: {deviation_heading} degrees for image {image_index+1} compared to {best_index+1}. estimated new heading: {estimated_new_heading} degrees. Actual heading change: {self.stored_headings[image_index]} degrees.")
        if image_index >= len(self.estimated_headings):
                # Extend the list with None or a default value up to the required index
                self.estimated_headings.extend([None] * (image_index + 1 - len(self.estimated_headings)))
        self.estimated_headings[image_index] = estimated_new_heading
        
        self.estimated_heading_deviations.append(np.abs(deviation_heading))
        
        # print(f"Estimated heading change: {estimated_new_heading} degrees for image {image_index+1} compared to {best_index+1}. Actual heading change: {self.stored_headings[image_index]} degrees. Deviation: {deviation_heading} degrees.")

            

    def find_best_match(self, image_index, image_space, grid_size=(5, 5), lower_percentile=20, upper_percentile=100-20):
        
        
        """Finds the best matching stored image for the given image index, using homography for rotation correction and grid-based ORB feature matching."""

        # Start timer to measure performance
        start_time = time.time()

        if len(self.stored_global_descriptors) == 0:
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
        linked_angles = []

        current_image = pull_image(image_index)
        current_grids = divide_into_grids(current_image, grid_size)

        for i in image_space:
            if i == image_index:
                continue
            
            stored_image = pull_image(i)
            kp_current = self.stored_global_keypoints[image_index]
            descriptors_current = self.stored_global_descriptors[image_index]

            kp_stored = self.stored_global_keypoints[i]
            descriptors_stored = self.stored_global_descriptors[i]

            # Match descriptors between the current image and the stored image
            detector_choice = 1 if self.global_detector_name == "ORB" else 2
            matches = self.global_matcher.find_matches(descriptors_current, descriptors_stored, kp_current, kp_stored, detector_choice, global_matcher_true=1)

             #knnMatch(descriptors_current, descriptors_stored, k=2)
            

            good_matches = []
            
            for match_pair in matches:
                if self.global_matcher_choice == 2: 
                    # Graph matcher returns singles (cv2.DMatch objects)
                    good_matches.append(match_pair)  # Handle single or list of matches
                else: 
                    # BFMatcher and FlannMatcher return pairs (tuple of two matches)
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.8 * n.distance:  # Lowe's ratio test, for rotations
                            good_matches.append(m)

            # initial rotating of image. 
            if len(good_matches) > 10:
                # Extract matched points
                src_pts = np.float32([kp_current[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_stored[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography to estimate transformation
                # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) xxx
                method_to_use = 0 # 0 is affine, 1 is homography
                M, amt_inliers = self.get_rotations(src_pts, dst_pts, method_to_use) 
                # M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC) # this angle is from the current image to the stored image. ie the image in question to the potential match. We want to reverse this.
                
                if amt_inliers < 30:
                    continue

                if M is not None:
                    # Rotate the stored image using the homography matrix
                    h, w = stored_image.shape[:2]
                    if method_to_use == 0:
                        rotated_image_stored = cv2.warpAffine(stored_image, M[:2, :], (w, h))
                    elif method_to_use == 1:
                        img = stored_image
                        height, width = img.shape[:2]

                        rotated_image_stored = cv2.warpAffine(img, M, (width, height)) #cv2.warpPerspective(stored_image, M, (w, h))

                    # Now divide both current and rotated stored image into grids
                    rotated_grids_stored = divide_into_grids(rotated_image_stored, grid_size)

                    total_good_matches = 0

                    # Perform grid-wise matching
                    for current_grid, stored_grid in zip(current_grids, rotated_grids_stored):
                        kp_current_grid, descriptors_current_grid = self.global_detector.get_keydes(current_grid)
                        #detectAndCompute(current_grid, None)
                        kp_stored_grid, descriptors_stored_grid = self.global_detector.get_keydes(stored_grid)
                        #.detectAndCompute(stored_grid, None)

                        if descriptors_current_grid is None or descriptors_stored_grid is None:
                            continue

                        # Match descriptors between grids using knnMatch
                        detector_choice = 1 if self.global_detector_name == "ORB" else 2
                        matches_grid = self.global_matcher.find_matches(descriptors_current_grid, descriptors_stored_grid, kp_current_grid, kp_stored_grid, detector_choice, global_matcher_true=1)

                        good_matches_grid = [] #  
                        for match_pair in matches_grid:
                            if self.global_matcher_choice == 2: 
                                good_matches_grid.append(match_pair)
                            else:                             
                                if len(match_pair) == 2:
                                    m, n = match_pair
                                    if m.distance < 0.999 * n.distance:  # Lowe's ratio test, per grid best match find. 0.999 worked. 
                                        good_matches_grid.append(m)
                        # Sum the good matches for each grid
                        total_good_matches += len(good_matches_grid) if len(good_matches_grid) < 50 else 50

                    # Score based on total good matches across all grids
                    score = total_good_matches

                    if score > max_corr_score:
                        max_corr_score = score
                        best_index = i
                        temp_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
                        linked_angles.append(temp_angle)


        # End timer and calculate time taken
        total_time = time.time() - start_time
        if len(linked_angles) > 0:
            # we need to think about how we deal with the angle and relativity. That is, we need to develop a standard initial image in the forward direction. 

            heading_change = -linked_angles[-1]
            # print(f"Heading from image {image_index+1} to {best_index+1} is {heading_change} degrees vs actual {-self.stored_headings[image_index]+self.stored_headings[best_index]} degrees.")
            self.compare_headings(image_index, best_index, heading_change)
            # print(f"Heading from image {image_index+1} to {best_index+1} is {heading_change} degrees.")
        


        # print(f"Best match for image {image_index+1} is image {best_index+1} with a total of {max_corr_score} good matches.")
        # print(f"Time taken: {total_time:.2f} seconds")
        
        return best_index



    def find_image_space(self, image_index):
        # this should look through the GPS values of all other images and return the indices, as an array, of all images with a corresponding GPS location within a radial distance of 100m of the current image.
        # this will be used to reduce the search space for the best match function.
        # the function will return an array of indices of images that are within the 100m radius of the current image.

        # get the GPS of the current image
        current_gps = self.stored_gps[image_index]#self.estimated_gps[-1] if len(self.estimated_gps) > 0 else self.stored_gps[-1]
        # Get the most recent estimation of GPS, if available, otherwise use the stored GPS - the last value before GPS loss. 
        # get the GPS of all other images
        image_space = []
        for i in range(len(self.stored_gps)):
            if i != image_index: # ensure, for Skripsie we dont infer from our own image.
                stored_gps = self.stored_gps[i]
                distance = np.linalg.norm(np.array(current_gps) - np.array(stored_gps)) # this takes the distance between both x,y coordinates and calculates the norm which is the distance between the two points. ie using pythagoras theorem.
                radial_distance_metres = 10000
                radial_distance_GPS = radial_distance_metres / 11139 #XXX
                if distance < radial_distance_GPS:
                    image_space.append(i)
        return image_space
     



    def clear_stored_data(self):
        """Clear stored images, keypoints, descriptors, and GPS data."""
        
        self.stored_gps = []
        self.estimated_gps_x_inference = []
        self.estimated_gps_y_inference = []
        self.actuals_x = []
        self.actuals_y = []
        
        

    def reset_all_data(self):
        """Clear stored images, keypoints, descriptors, and GPS data."""
        # print("Resetting all data.")
        self.stored_image_count = 0
        self.stored_global_keypoints = []
        self.stored_global_descriptors = []
        self.stored_local_keypoints = []
        self.stored_local_descriptors = []
        self.stored_gps = []
        self.inferred_factor_x = 1
        self.inferred_factor_y = 1
        self.actuals_x = []
        self.actuals_y = []
        self.global_matcher = None
        self.local_matcher = None
        self.global_matcher_choice = 0
        self.neural_net_on = False
        self.global_matching_technique = 0
        self.local_detector_name = ""
        self.global_detector_name = ""
        self.time_best_match_function = 0
        self.stored_headings = []
        self.estimated_headings = []
        self.estimated_gps = []
        self.estimated_gps_x_inference = []
        self.estimated_gps_y_inference = []
        self.estimated_heading_deviations = []
        self.estimated_heading_deviations = []
        self.estimated_gps_x = []
        self.estimated_gps_y = []
        self.norm_GPS_error = []





    

    def add_image(self, kernel_to_test, index, directory):

        image_path = os.path.join(directory, f'{index}.jpg')
        gps_path = os.path.join(directory, f'{index}.txt')
        image = cv2.imread(image_path)  # Read the image in color
        gps_coordinates, heading = parse_gps(gps_path)
        """Add an image and its GPS coordinates to the stored list."""
        cropped_image = crop_image(image, kernel_to_test)
        
        # Convert to grayscale for detectors if not already
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
        keypoints, descriptors = self.global_detector.get_keydes(gray_image)#self.global_detector.detectAndCompute(gray_image, None) # have both for local and global 

        # Check if descriptors are None
        if descriptors is None:
            print(f"Warning: No descriptors found for one image. Skipping.")
            return

        self.stored_global_keypoints.append(keypoints)
        self.stored_global_descriptors.append(descriptors)
        self.stored_gps.append(gps_coordinates)
        self.stored_headings.append(heading)
        self.stored_image_count += 1

        if self.global_detector_name == self.local_detector_name:
            self.stored_local_keypoints.append(keypoints)
            self.stored_local_descriptors.append(descriptors)
        else:
            # Extract local keypoints and descriptors
            local_keypoints, local_descriptors = self.local_detector.get_keydes(gray_image)
            if local_descriptors is None:
                print(f"Warning: No local descriptors found for one image. Skipping.") 
                return
            self.stored_local_keypoints.append(local_keypoints)
            self.stored_local_descriptors.append(local_descriptors)


    def compute_linear_regression_factors(self):
        """Compute the linear regression factors for both x and y based on the stored estimates and actual values."""
        
        # Prepare data for linear regression
        x_estimates = np.array(self.estimated_gps_x_inference).reshape(-1, 1)
        y_estimates = np.array(self.estimated_gps_y_inference).reshape(-1, 1)
        
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
            print(f"Linear regression inferred factor y: {inferred_factor_y} \n")

            # Update the inferred factors
            self.inferred_factor_x = inferred_factor_x
            self.inferred_factor_y = inferred_factor_y
        else:
            print("Not enough data points to perform linear regression.")

    def get_pixel_shifts(self, keypoints1, descriptors1, keypoints2, descriptors2, rotated_image, i, lower_percentile=20, upper_percentile=80):
        timeA = time.time()
        
        detector_choice = (1 if self.local_detector_name == "ORB" else 2)
        matches = self.local_matcher.find_matches(descriptors1, descriptors2, keypoints1, keypoints2, detector_choice, global_matcher_true=0)

        match_ratio = 0.8
        good_matches = []
        
        for match_pair in matches:
                if self.local_matcher_choice == 2:  # Graph matcher returns singles
                    good_matches.append(match_pair)  # Handle single or list of matches
                else:             
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < match_ratio * n.distance:
                            good_matches.append(m)

        if len(good_matches) < 4:
            print("Warning: Less than 4 matches found.")
            return None, None, None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        shifts1 = dst_pts - src_pts
        # infer x and y for shifts1
        translation_x1 = np.mean(shifts1[0]) # bare bones method
        translation_y1 = np.mean(shifts1[1])

        CURRENT_IMAGE = pull_image(i)


        shift, response = cv2.phaseCorrelate(np.float32(CURRENT_IMAGE), np.float32(rotated_image))
        
        translation_x, translation_y= shift
        
        # invert translation x
        translation_x = translation_x

        #print the difference in pixel values for both x and y of both estimation methods. ie translation_x1-translation_x, translation_y1-translation_y
        print("METHOD_1: ", translation_x1, translation_y1)
        print("METHOD_2: ", translation_x, translation_y)

        print(f"Phase Correlation estimated translation: {translation_x,translation_y} for image {i+1} vs whatever the match")
        
        # shifts = dst_pts - src_pts
        # M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25)
        # translation_x = M[1, 2]  # Extract the translation along x
        # translation_y = M[0, 2]  # Extract the translation along y
         
        # timeB = time.time() - timeA
        # print(f"Time taken for pixel shift calculation: {timeB:.2f} seconds")
        # return shifts if len(shifts) != 0 else None, translation_x, translation_y
        return translation_x, translation_y

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

    def analyze_matches(self, lower_percentile, bool_infer_factor, num_images_analyze):
        deviation_norms_x = []
        deviation_norms_y = []
        rotations_arr = []
        ''' 
        first we add X images to infer. That should go on until we meet the required count
        # then we add Y images if GPS is available. 
        # then we add Z images if GPS is not available. 
        # in the test data we have a limited amount of images. 

        # in our test case:
        # we have 13 images.
        # the first 6 infers the factors. 
        # we then add the remaining 7 images.
        # we then analyze backwards from the 13th image, only looking at prior images OR:
        # we start with the images after the 13th which are distinctly different as they are on the flight back. We then start from 13+1 -> end of images e.g., 26. then range_im = self.stored_image_count + 1 (=14) -> 26

        '''

        range_im = num_images_analyze
        # if we have lost GPS, we stream the latest image which is the 13th+1th image.
        
        for i in reversed(range(1, range_im)):  # iterate through all images in reverse order
            best_index = -1   
            #best_index = i - 1
            image_space = self.find_image_space(i)
            timeC = time.time()
            if self.global_matching_technique < 3:
                best_index = self.find_best_match(i, image_space)
            else: 
                best_index = self.find_best_match_multimethod(i, image_space, self.global_matcher_choice
                )
            timeD = time.time() - timeC
            print(f"Time taken for best match function: {timeD:.2f} seconds")

            if best_index != -1:
                #print(f"Best match for image {i+1} is image {best_index+1} with undefined good matches.")
                #best_index = i - 1
                angle = self.estimated_headings[i] - self.stored_headings[best_index] if len(self.estimated_headings) > 0 else None
                angle = -angle if angle is not None else None ### XXX
                # print(f"Estimated heading for image {i}: {self.estimated_headings[i]} degrees.")

                
                # test if angle is not None
                if angle is not None:

                    # find actual xy pixel changes from GPS for testing
                    actual_gps_diff = np.array(self.stored_gps[i]) - np.array(self.stored_gps[best_index])
                    actual_gps_diff_meters = actual_gps_diff * 111139  # Convert degrees to meters
                    actual_pixel_change_x_m = actual_gps_diff_meters[0] 
                    actual_pixel_change_y_m = actual_gps_diff_meters[1] 
                    

                    initial_stored_image = pull_image(best_index)
                    rotated_image = self.rotate_image(initial_stored_image, angle)
                    rotated_keypoints, rotated_descriptors = self.local_detector.get_keydes(rotated_image)
                    rotations_arr.append(angle)  # Append rotation to the array
                    CURRENT_image = pull_image(i)
                    
                    # translation_x, translation_y = self.get_pixel_shifts(
                    #     self.stored_local_keypoints[i], self.stored_local_descriptors[i], 
                    #     rotated_keypoints, rotated_descriptors, rotated_image, i
                    # )
                    if best_index<i:
                        translation_y, translation_x = estimate_shift_from_points(CURRENT_image, rotated_image) 

                    if translation_y>0 and actual_gps_diff_meters[1]<0 or translation_y<0 and actual_gps_diff_meters[1]>0:
                        print(f"Irregular sign at image {i+1} wrt image {best_index+1}")
                    # translation_y = -translation_y  # Invert y translation
                    # translation_x = -translation_x  # Invert x t  ranslation
                    


                    translation_y = -translation_y #shifts[:, 0]
                    print(f"Estimated translation: ({translation_x}, {translation_y})")
                    pixel_changes_x = translation_x #shifts[:, 0]
                    pixel_changes_y = translation_y #shifts[:, 1]
                    mean_pixel_changes_x = np.mean(pixel_changes_x)
                    mean_pixel_changes_y = np.mean(pixel_changes_y)



                    # mean_pixel_changes = np.array([mean_pixel_changes_x, mean_pixel_changes_y])
                    # # rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                    # #         [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
                    # # estimated_gps_diff_rotated = np.dot(rotation_matrix, mean_pixel_changes)
                    # # mean_pixel_changes_x = estimated_gps_diff_rotated[0]
                    # # mean_pixel_changes_y = estimated_gps_diff_rotated[1]
                    # mean_pixel_changes_x = mean_pixel_changes[0]
                    # mean_pixel_changes_y = mean_pixel_changes[1]

                    # print(f"Mean vs actual X pixel change: {mean_pixel_changes_x} vs {actual_pixel_change_x_m} meters for image {i+1}")
                    # print(f"Mean vs actual Y pixel change: {mean_pixel_changes_y} vs {actual_pixel_change_y_m} meters for image {i+1}")

        

                                      
                    if bool_infer_factor and actual_pixel_change_x_m != 0 and actual_pixel_change_y_m != 0:
                        # Estimate linear regression factors with mse function. These are difference values not actual coords. 
                        self.estimated_gps_x_inference.append(mean_pixel_changes_x)
                        self.estimated_gps_y_inference.append(mean_pixel_changes_y)
                        self.actuals_x.append(actual_pixel_change_x_m)
                        self.actuals_y.append(actual_pixel_change_y_m)


                    # ESTIMATIONS
                    mean_pixel_changes_x_m = mean_pixel_changes_x * self.inferred_factor_x 
                    mean_pixel_changes_y_m = mean_pixel_changes_y * self.inferred_factor_y


                    # NEW GPS
                    new_lon = self.stored_gps[best_index][0] + mean_pixel_changes_x_m / 111139
                    new_lat = self.stored_gps[best_index][1] + mean_pixel_changes_y_m / 111139
                    if not bool_infer_factor:
                        self.estimated_gps.append((new_lon, new_lat))


                    # Calculate deviation from actual pixel changes for testing
                    deviation_x_meters = mean_pixel_changes_x_m - actual_pixel_change_x_m
                    deviation_y_meters = mean_pixel_changes_y_m - actual_pixel_change_y_m
                    deviation_norms_x.append((deviation_x_meters))
                    deviation_norms_y.append(np.abs(deviation_y_meters))
                    print(f"Deviations: {deviation_x_meters} meters (x), {deviation_y_meters} meters (y) for image {i+1}")


        # find the norm of the deviations for x and y, and calc its norm
        mean_x_dev = np.mean(deviation_norms_x)
        mean_y_dev = np.mean(deviation_norms_y)
        total_normalized_gps_error = np.linalg.norm([mean_x_dev, mean_y_dev])
        if not bool_infer_factor:
            self.norm_GPS_error.append(total_normalized_gps_error)




def parse_gps(file_path):
    """Parse GPS coordinates from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lon_str = lines[1].strip()  # (x/lon is second in notepad)
        lat_str = lines[0].strip()  # lat is y (first line)
        heading_str = lines[2].strip()  # Heading is the third line

        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_str)
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_str)
        lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
        lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)
        heading = float(heading_str.split(": ")[1])
        return (lon, -lat), heading  # invert y-axis as it's defined as South First. return x first then y

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

def crop_image(img, kernel_to_test):
    """Crop the top and bottom 10% of the image."""

    height, width = img.shape[:2]
    crop_size = int(height * 0.1)  # 10% of the height
    # cropped_image = cv2.GaussianBlur(cropped_image, (kernel_to_test, kernel_to_test), 0)  # Denoise
    return img[crop_size:height-crop_size, :]  # Crop top and bottom

def pull_image(index):
    """Pull an image from from the directory with name index.jpg"""
    directory = './GoogleEarth/SET2'
    image_path = os.path.join(directory, f'{index+1}.jpg')
    
    # read in grey 
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
    # read in grey
    kernel = 3
    image = cv2.imread(image_path)  # Read the image in
    cropped_image = crop_image(image, 3)
        
        # Convert to grayscale for detectors if not already
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
    return gray_image

def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0
    if dir in ['S', 'W']:
        decimal = -decimal
    return decimal

def phase_correlation_rotation(img1, img2):
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    # Apply Canny edge detection
    edges1 = cv2.Canny(gray1, 0, 300)
    edges2 = cv2.Canny(gray2, 0, 300)

    # Use Hough Line Transform to detect lines
    lines1 = cv2.HoughLines(edges1, 1, np.pi / 180, 100)
    lines2 = cv2.HoughLines(edges2, 1, np.pi / 180, 100)

    # Compute the average angle of the lines in both images
    def average_angle(lines):
        angles = [line[0][1] for line in lines]
        return np.mean(angles) * (180 / np.pi)  # Convert to degrees

    angle1 = average_angle(lines1)
    angle2 = average_angle(lines2)

    # The rotation angle is the difference between the two
    rotation_angle = angle2 - angle1
    return rotation_angle

def print_choices(global_detector_choice,global_matcher_choice, global_matching_technique, local_detector_choice, local_matcher_choice):
    # In the global matching stage I have: preprocessing stage to normalize rotations: preproc_local_alg_detector, preproc_local_alg_matcher; Then I do the global matching stage: with either a global_matcher or a local_matcher - defined generally as global_matcher. Finally, I use the matched image to perform the local matching stage: with a local_detector and a local_matcher.


    printable_preproc_glob_detector = ""
    printable_preproc_glob_matcher = ""
    printable_global_matching_technique = ""
    printable_loc_detector = ""
    printable_loc_matcher = ""


    if global_detector_choice == 1:
        printable_preproc_glob_detector = "ORB"
    elif global_detector_choice == 2:
        printable_preproc_glob_detector = "AKAZE"

    if global_matcher_choice == 0:
        printable_preproc_glob_matcher = "BF"
    elif global_matcher_choice == 1:
        printable_preproc_glob_matcher = "FLANN"
    elif global_matcher_choice == 2:
        printable_preproc_glob_matcher = "GRAPH"


    
    if global_matching_technique <3: # ie if im using grid divide then it is for eg "ORB" and "BF"
        printable_global_matching_technique = "Same as Global Matcher"
    elif global_matching_technique == 3:
        printable_global_matching_technique = "Cross Correlation"
    elif global_matching_technique == 4:
        printable_global_matching_technique = "Histogram"
    elif global_matching_technique == 5:
        printable_global_matching_technique = "SSIM"
    elif global_matching_technique == 6:
        printable_global_matching_technique = "Hash"


    if local_detector_choice == 1:
        printable_loc_detector = "ORB"
    elif local_detector_choice == 2:
        printable_loc_detector = "AKAZE"

    if local_matcher_choice == 0:
        printable_loc_matcher = "BF"
    elif local_matcher_choice == 1:
        printable_loc_matcher = "FLANN"
    elif local_matcher_choice == 2:
        printable_loc_matcher = "GRAPH"


    string_print = f"Preprocessing Global Detector: {printable_preproc_glob_detector}, Preprocessing Global Matcher: {printable_preproc_glob_matcher}, Global Matching Technique: {printable_global_matching_technique}, Local Detector: {printable_loc_detector}, Local Matcher: {printable_loc_matcher}"
    print(string_print)
    return string_print

    # add all of this as a string for return

def append_to_file(filename="OVERWRITE_ME", *data):
    directory = os.path.join("GoogleEarth", "RESULTS")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'a') as file:
        for line in data:
            file.write(line + '\n')

def main():
    


# super_ ASD

    global_detector_choice = 2  # # Set 1 for ORB, 2 for AKAZE. These are used for rotations. 
    global_matcher_choice = 0  # Set 0 for BFMatcher, 1 for FlannMatcher, 2 for GraphMatcher/ This and above are for global rotational matching. 
    global_matcher_technique = 0  # Set 0 for Pass Through grid counter, 3 for correlation, 4 for histogram, 5 for SSIM, 6 for Hash.
    # code range [1-2], [0-2], [0,3,4,5,6]

    local_detector_choice = 2  # Set 1 for ORB, 2 for AKAZE.
    local_matcher_choice = 0  # Set 0 for BFMatcher, 1 for FlannMatcher, 2 for GraphMatcher

    # akaze and bf is the best for rotational matching. 

    # Best combo: global/rotation:affine,  AKAZE and BF, technique: histogram; local: akaze and flann. 

    directory = './GoogleEarth/SET1'
    num_images = 9
    inference_images = 7
    lower_percentiles = [20]
    variable_to_iterate = [0, 1] # put 0 back xxx 0,3,4,5,6. DONT DO 6. 
    iteration_count = 0

    global_detector_arr = [1,2]
    global_matcher_arr = [0,1,2]
    global_matcher_technique_arr = [0,3,4,5]
    local_detector_arr = [1,2]
    local_matcher_arr = [0,1,2]
    if 1==1:
    # for global_detector_choice in global_detector_arr:
    #     for global_matcher_choice in global_matcher_arr:
    #         for global_matcher_technique in global_matcher_technique_arr:
    #             for local_detector_choice in local_detector_arr:
    #                 for local_matcher_choice in local_matcher_arr:
                        # if the navigator object exists

                        # if the navigator object exists
                        if 'navigator' in locals():
                            navigator.reset_all_data()
                        main_start_time = time.time()
                        navigator = UAVNavigator(global_detector_choice, local_detector_choice , global_matcher_choice, local_matcher_choice, global_matcher_technique) # INITIALIZATION
                        _ = print_choices(global_detector_choice,global_matcher_choice, global_matcher_technique, local_detector_choice, local_matcher_choice)
                        kernel = 3 # Kernel size for Gaussian blur
                        
                        iteration_count += 1
                        for lower_percentile in lower_percentiles:
                            navigator.clear_stored_data()  # Clear stored data before each kernel test

                            # Step 1: Add images and infer factors
                            for i in range(1, inference_images + 1): # stops at inference_images = 6
                                navigator.add_image(kernel, i, directory)

                            # analyse Deviations to get inferred factors. BOOL_INFER_FACTOR = TRUE
                            navigator.analyze_matches(lower_percentile, True, inference_images)
                            navigator.compute_linear_regression_factors()
                            # print("INFERRED FACTORS:", navigator.inferred_factor_x, navigator.inferred_factor_y)

                            for i in range(inference_images+1, num_images + 1):
                                navigator.add_image(kernel, i, directory)

                            not_stream_images = 0
                            # BOOL INFER FACTOR = FALSE. num_images_analyze = 13
                            navigator.analyze_matches(lower_percentile, False, num_images)
                            string_params = print_choices(global_detector_choice,global_matcher_choice, global_matcher_technique, local_detector_choice, local_matcher_choice)
                            string_GPS_error = f"Mean normalized GPS error: {navigator.norm_GPS_error}"
                            string_heading_error = f"Mean Heading Error: {np.mean(navigator.estimated_heading_deviations)}"
                            print(string_params, '\n', string_GPS_error, '\n', string_heading_error)
                            append_to_file("results.txt", string_params, string_GPS_error, string_heading_error, "\n\n")
                            
                            # print the mean, max and min of the navigator, runtime rotational normalizer runtime_rotational_estimator
                            # print(f"Mean Rotational Normalizer Time: {np.mean(navigator.runtime_rotational_estimator)} , Max Rotational Normalizer Time: {np.max(navigator.runtime_rotational_estimator)}, Min Rotational Normalizer Time: {np.min(navigator.runtime_rotational_estimator)} ms, median: {np.median(navigator.runtime_rotational_estimator)}") if 


                            elapsed_time = time.time() - main_start_time
                            print(f"Time taken to execute The Method: {elapsed_time:.4f} seconds")
                            print(f'End of Iteration: {iteration_count}')
                            
                            # print the mean, max and min of the navigator, runtime_rotational_estimator        
                    
                    
    # ANY code outside of all iterations. 
    


if __name__ == "__main__":
    main() 
