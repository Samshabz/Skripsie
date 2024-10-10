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
from Feature_Extractors import set_feature_extractor, set_neural_feature_extractor
from shiftdetectors import get_src_shifts, get_neural_src_pts
from translation_methods import *
import gc



class UAVNavigator:
    def __init__(self, global_detector_choice, local_detector_choice, global_matcher_choice, local_matcher_choice, global_matching_technique, dataset_name, global_detector_threshold=0.001, local_detector_threshold=0.001):
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
        self.dataset_name = ""

        


        # angles 
        self.stored_headings = []
        self.estimated_headings = []
        self.estimated_heading_deviations = []
        
        # actual GPS
        self.actual_GPS_deviation = []

        # estimated GPS
        self.estimated_gps = []
        self.estimated_gps_x_inference = []
        self.estimated_gps_y_inference = []
        self.norm_GPS_error = []
        

        # Neural Network
        self.stored_feats = []


        # Stability of translation estimations DEBUG
        self.mag1_stability_arr = []
        self.mag2_stability_arr = []
        self.mag3_stability_arr = []
        self.mag4_stability_arr = []
        self.mag5_stability_arr = []
        self.string_stability = []

        # Loss of GPS
        self.global_loss_keypoints = []
        self.global_loss_descriptors = []
        self.local_loss_keypoints = []
        self.local_loss_descriptors = []
        self.local_loss_feats   = []

        self.same_detector_threshold = False

        # DEBUG 
        self.keypoint_time = 0 
        self.keypoint_iterations = 0
        self.len_matches_arr = []
        self.glob_mat_len_arr = []
        self.loc_mat_len_arr = []



        if local_detector_choice == 3:
            self.neural_net_on = True

        if global_detector_threshold == local_detector_threshold:
            self.same_detector_threshold = True
        
        # datasets
        self.dataset_name = dataset_name # pass as string
        self.directory = './GoogleEarth/DATASETS/{}'.format(self.dataset_name)


        # timing 
        self.time_best_match_function = 0
        self.runtime_rotational_estimator = []

        if global_detector_choice == 1:
            self.global_detector_name = "ORB"
            self.global_detector = set_feature_extractor(global_detector_choice, global_detector_threshold)
        elif global_detector_choice == 2: 
            
            self.global_detector_name = "AKAZE"
            self.global_detector = set_feature_extractor(global_detector_choice, global_detector_threshold)


        if local_detector_choice == 1:                
            self.local_detector_name = "ORB"
            self.local_detector = set_feature_extractor(local_detector_choice, local_detector_threshold)
        elif local_detector_choice == 2: 
            self.local_detector_name = "AKAZE"
            self.local_detector = set_feature_extractor(local_detector_choice, local_detector_threshold)
            

        elif local_detector_choice == 3:
            if self.neural_net_on:
                self.local_detector_name = "SuperPoint"
                self.local_detector = set_neural_feature_extractor()
        

        # Initialize the detector based on choice
        else:
            raise ValueError("Invalid detector choice! Please choose 1 for ORB, 2 for AKAZE, 3 for   SuperPoint.")


        globchoice = "bf_matcher" if global_matcher_choice == 0 else "flann_matcher" if global_matcher_choice == 1 else "graph_matcher" if global_matcher_choice == 2 else "histogram_matcher" if global_matcher_choice == 3 else "SSIM_matcher"
        self.global_matcher_choice = global_matcher_choice
        locchoice = "bf_matcher" if local_matcher_choice == 0 else "flann_matcher" if local_matcher_choice == 1 else "graph_matcher"
        self.local_matcher_choice = local_matcher_choice


        self.global_matcher = set_matcher(globchoice) # 0 for BFMatcher, 1 for FlannMatcher, 2 for graph matcher. 

        self.local_matcher = set_matcher(locchoice) 


        self.global_matching_technique = global_matching_technique  

    def get_rotations(self, src_pts, dst_pts, method_to_use):
        homography_threshold = 25 if self.global_detector_name == "ORB" else 0.5
        new_params_ransac = 25 if self.global_detector_name == "AKAZE" else 0.5

        M = None
        mask = None
        rotation_angle = 0
        if method_to_use == 0:
            # This needs to actually be tuned. 
            M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
            A = M[:2, :2]
            
            # Perform SVD to decompose A into scaling and rotation (no shear)
            U, S, Vt = np.linalg.svd(A)
            
            # Rotation matrix is U
            rotation_angle = np.arctan2(U[1, 0], U[0, 0]) * (180 / np.pi)
            
            # # Scaling factors are in S
            # scale_x = S[0]
            # scale_y = S[1]

            
        elif method_to_use == 1:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_threshold)
            M = M / M[2, 2]
        
            # The top-left 2x2 submatrix contains the rotational and scaling info
            rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
            scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)

        elif method_to_use == 2:
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is None:
            raise ValueError("Transformation matrix estimation failed.")

        return M, np.sum(mask), rotation_angle
    
    def extract_rotation_and_scale(M, method):

        if method == 'affine':
            # For affine, the top-left 2x2 submatrix contains the rotation and scaling info
            rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
            scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)

        elif method == 'homography':
            # Normalize homography matrix to remove scale effect
            M = M / M[2, 2]
            
            # The top-left 2x2 submatrix contains the rotational and scaling info
            rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
            scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
        
        else:
            raise ValueError("Method must be either 'affine' or 'homography'")

        # Return the rotation angle and the average scaling factor (uniform scaling)
        scale = (scale_x + scale_y) / 2
        return rotation_angle, scale



    def get_src_dst_pts(self, kp1, kp2, des1, des2, Lowes_thresh = 0.8, global_matcher_true = True ): # XXX
            if des1 is None or des2 is None:
                raise ValueError("One or both descriptors are None.")

            # Match keypoints using BFMatcher
            
            matches = self.global_matcher.find_matches(des1, des2) if global_matcher_true else self.local_matcher.find_matches(des1, des2) 
            if self.local_detector_name == "ORB" and not global_matcher_true and self.dataset_name != "DATSETSAND":
                Lowes_thresh -= 0.3
            good_matches = []
            while len(good_matches) < 500:
                for match_pair in matches:
                    if self.global_matcher_choice == 2: 
                        # Graph matcher returns singles (cv2.DMatch objects)
                        good_matches.append(match_pair)  # Handle single or list of matches
                    else: 
                        # BFMatcher and FlannMatcher return pairs (tuple of two matches)
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < Lowes_thresh * n.distance:  # Lowe's ratio test, for rotations
                                good_matches.append(m)
                Lowes_thresh += 0.025

            # the above generally ensures the 500 most non-amiguous matches are returned. At least with ORB, having a dynamically increasing ambiguity threshold seems extremely beneficial.
            # the following limits the matches based on similarity to its match
           
            if global_matcher_true:
                global_kp_allowed = 300
                if len(good_matches) > global_kp_allowed: # XXX optimize this
                    sorted_matches = sorted(good_matches, key=lambda x: x.distance)
                    if self.global_detector_name == "AKAZE":
                        good_matches = sorted_matches[:global_kp_allowed]
                    elif self.global_detector_name == "ORB":
                        good_matches = sorted_matches[:global_kp_allowed*5]
            elif not global_matcher_true:
                local_kp_allowed = 1000
                if len(good_matches) > local_kp_allowed: 
                    sorted_matches = sorted(good_matches, key=lambda x: x.distance)
                    #  apply the same logic for local detector - improved results for transformations. 
                    if self.local_detector_name == "AKAZE":
                        good_matches = sorted_matches[:local_kp_allowed]
                    elif self.local_detector_name == "ORB":
                        good_matches = sorted_matches[:local_kp_allowed*2]
            if global_matcher_true:
                self.glob_mat_len_arr.append(len(good_matches)) 
            elif not global_matcher_true:
                self.loc_mat_len_arr.append(len(good_matches))
            # Extract matched keypoints 
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            
            return src_pts, dst_pts
    
    

    



    def find_best_match_multimethod(self, image_index, image_space):
        # print("Finding best match using multiple methods.")
        def cross_correlate(image1, image2):
            """Performs cross-correlation between two full images."""
            result = cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)
            _, max_corr, _, _ = cv2.minMaxLoc(result)
            return max_corr
        """Finds the best matching stored image for the given image index using cross-correlation with rotational normalization."""


        best_index = -1
        max_corr_score = -np.inf  # Initial maximum correlation score
        linked_angles = []

        inference_img = self.pull_image(image_index, self.directory, bool_rotate=False)  # Load the current image
        kp_inf, des_inf = self.global_detector.get_keydes(inference_img)
        

        
        for ref in image_space:
            
            if ref == image_index:  # Can change this to if i >= image_index to only infer from lower indices
                continue  
            ref_img = self.pull_image(ref, self.directory, bool_rotate=True)
            kp_stored = self.stored_global_keypoints[ref]
            descriptors_stored = self.stored_global_descriptors[ref]
            Lowes_ratio = 0.8 if self.global_detector_name == "AKAZE" else 0.7
            src_pts, dst_pts = self.get_src_dst_pts(kp_inf, kp_stored, des_inf, descriptors_stored, Lowes_ratio)

            method_to_use = 0 # 0 is affine, 1 is homography
            M, amt_inliers, _ = self.get_rotations(src_pts, dst_pts, method_to_use) # inf = src = img_index, dst = ref = i = stored
            # print(f"AMT INLIERS: {amt_inliers}\n")
            
            if amt_inliers < 80 or M is None:
                continue
            

            h, w = ref_img.shape[:2]
            if method_to_use == 0:
                rot_ref = cv2.warpAffine(ref_img, M[:2, :], (w, h))
            elif method_to_use == 1 or method_to_use == 2:
                rot_ref = cv2.warpAffine(ref_img, M,  (w, h))

            # Perform cross-correlation directly on the full image
            if self.global_matching_technique == 3:
                score = cross_correlate(inference_img, rot_ref)


            elif self.global_matching_technique == 4 or 1==1:
                score = self.compare_histograms(inference_img, rot_ref)
                

            elif self.global_matching_technique == 5:
                score = self.compare_ssim(inference_img, rot_ref)
                


            elif self.global_matching_technique == 6:
                score = self.compare_hash(inference_img, rot_ref)
                
            if score > max_corr_score:
                max_corr_score = score
                best_index = ref
                temp_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
                linked_angles.append(temp_angle)
                


        heading_change = 0
        if len(linked_angles) > 0:
            heading_change = -linked_angles[-1]
            self.compare_headings(image_index, best_index, heading_change)
        
        return best_index, -heading_change

    def pull_image(self, index, directory, bool_rotate=False, use_estimate=False):
        """Pull an image from from the directory with name index.jpg"""
        image_path = os.path.join(directory, f'{index+1}.jpg')
        
        # read in grey 
        #image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
        # read in grey
        image = cv2.imread(image_path)  # Read the image in
        cropped_image = crop_image(image)
            
            # Convert to grayscale for detectors if not already
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
        if bool_rotate:
            gray_image = rotate_image(gray_image, self.stored_headings[index]) if not use_estimate else rotate_image(gray_image, self.estimated_headings[index])
        return gray_image

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
    
    def compare_headings(self, inference_index, best_index, estimated_heading_change):
        # since both images have been normalized, this no longer is applicable XXX  
        # Normalize angles to be within 0 to 360 or -180 to 180 range as needed
        def normalize_angle(angle):
            angle = angle % 360  # Normalize to [0, 360)
            if angle > 180:  # Normalize to [-180, 180)
                angle -= 360
            return angle

        # Calculate the new heading, then normalize it
        estimated_heading_change = normalize_angle(estimated_heading_change)
        self.stored_headings[best_index] = normalize_angle(self.stored_headings[best_index]) # this we have as we are using it to infer the rotation. 
        self.stored_headings[inference_index] = normalize_angle(self.stored_headings[inference_index]) # this is only used for error comparison. 

        estimated_new_heading = normalize_angle(estimated_heading_change)
        

        deviation_heading = self.stored_headings[inference_index] - estimated_new_heading
        deviation_heading = normalize_angle(deviation_heading)
        
        if inference_index >= len(self.estimated_headings):
                # Extend the list with None or a default value up to the required index
                self.estimated_headings.extend([None] * (inference_index + 1 - len(self.estimated_headings)))
        self.estimated_headings[inference_index] = estimated_new_heading
        
        self.estimated_heading_deviations.append(np.abs(deviation_heading))

    def find_good_matches(self, matches, Lowe_thresh):
        good_matches = []
        for match_pair in matches:
            if self.global_matcher_choice == 2: # XXX 
                # Graph matcher returns singles (cv2.DMatch objects)
                good_matches.append(match_pair)  # Handle single or list of matches
            else: 
                # BFMatcher and FlannMatcher return pairs (tuple of two matches)
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < Lowe_thresh * n.distance:  # Lowe's ratio test, for rotations
                        good_matches.append(m)
        return good_matches



    def find_best_match(self, image_index, image_space, grid_size=(4, 4)):

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





        inference_img = self.pull_image(image_index, self.directory, bool_rotate=False)  # Load the current image
        kp_inf, des_inf = self.global_detector.get_keydes(inference_img)
        
        

        current_grids = divide_into_grids(inference_img, grid_size)

        for ref in image_space:
            if ref == image_index: 
                continue
            
            ref_img = self.pull_image(ref, self.directory, bool_rotate=True)
            kp_stored = self.stored_global_keypoints[ref]
            descriptors_stored = self.stored_global_descriptors[ref]

            detector_choice = 1 if self.global_detector_name == "ORB" else 2
            Lowes_Ratio = 0.8 if self.global_detector_name == "AKAZE" else 0.7
            src_pts, dst_pts = self.get_src_dst_pts(kp_inf, kp_stored, des_inf, descriptors_stored, Lowes_Ratio)
 

            # initial rotating of image. 
            if len(src_pts) > 10:
                # Extract matched points

                method_to_use = 0 # 0 is affine, 1 is homography # 2 is partial affine
                M, amt_inliers, _ = self.get_rotations(src_pts, dst_pts, method_to_use) 
                # print(f"AMT INLIERSBM: {amt_inliers}\n")
                
                if amt_inliers < 80 or M is None:
                    continue

                # Rotate the stored image using the homography matrix
                h, w = ref_img.shape[:2]
                if method_to_use == 0:
                    rotated_image_stored = cv2.warpAffine(ref_img, M[:2, :], (w, h))
                elif method_to_use == 1 or method_to_use == 2:
                    rotated_image_stored = cv2.warpAffine(ref_img, M,  (w, h))

                # Now divide both current and rotated stored image into grids
                rotated_grids_stored = divide_into_grids(rotated_image_stored, grid_size)

                total_good_matches = 0

                # Perform grid-wise matching
                for current_grid, stored_grid in zip(current_grids, rotated_grids_stored):
                    kp_current_grid, descriptors_current_grid = self.global_detector.get_keydes(current_grid)
                    kp_stored_grid, descriptors_stored_grid = self.global_detector.get_keydes(stored_grid)

                    if descriptors_current_grid is None or descriptors_stored_grid is None:
                        continue

                    # Match descriptors between grids using knnMatch
                    detector_choice = 1 if self.global_detector_name == "ORB" else 2
                    matches_grid = self.global_matcher.find_matches(descriptors_current_grid, descriptors_stored_grid, kp_current_grid, kp_stored_grid, detector_choice, global_matcher_true=1)

                    good_matches_grid = self.find_good_matches(matches_grid, 0.8)

                    total_good_matches += len(good_matches_grid) if len(good_matches_grid) < 20 else 20

                # Score based on total good matches across all grids
                score = total_good_matches

                if score > max_corr_score:
                    max_corr_score = score
                    best_index = ref
                    temp_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
                    # print(f"TEMP ANGLE1: {temp_angle}\n")
                    linked_angles.append(temp_angle)


        heading_change = 0
        if len(linked_angles) > 0:
            heading_change = -linked_angles[-1]
            self.compare_headings(image_index, best_index, heading_change)

        return best_index, -heading_change



    def find_image_space(self, image_index):
        # this should look through the GPS values of all other images and return the indices, as an array, of all images with a corresponding GPS location within a radial distance of 100m of the current image.
        # this will be used to reduce the search space for the best match function.
        # the function will return an array of indices of images that are within the 100m radius of the current image.

        # get the GPS of the current image
        current_gps = self.stored_gps[image_index]#self.estimated_gps[-1] if len(self.estimated_gps) > 0 else self.stored_gps[-1]
        # Get the most recent estimation of GPS, if available, otherwise use the stored GPS - the last value before GPS loss. 
        # get the GPS of all other images
        image_space = []
        iterative_radius = 5000
        while len(image_space)<5:
            for i in range(len(self.stored_gps)):
                if i != image_index: # ensure, for Skripsie we dont infer from our own image.
                    stored_gps = self.stored_gps[i]
                    distance = np.linalg.norm(np.array(current_gps) - np.array(stored_gps)) # this takes the distance between both x,y coordinates and calculates the norm which is the distance between the two points. ie using pythagoras theorem.
                    radial_distance_metres = 5000
                    radial_distance_GPS = radial_distance_metres / 11139 #XXX
                    if distance < radial_distance_GPS:
                        image_space.append(i)
            iterative_radius += 2500
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
        self.actual_GPS_deviation = []
        self.stored_feats = []
        self.runtime_rotational_estimator = []
        
        
    

    def add_image(self, index, directory):

        image_path = os.path.join(directory, f'{index}.jpg')
        gps_path = os.path.join(directory, f'{index}.txt')
        image = cv2.imread(image_path)  # Read the image in color
        gps_coordinates, heading = parse_gps(gps_path)
        """Add an image and its GPS coordinates to the stored list."""
        cropped_image = crop_image(image)
        self.stored_gps.append(gps_coordinates)
        self.stored_headings.append(heading)
        self.stored_image_count += 1
        
        # Convert to grayscale for detectors if not already
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image

        gray_image = rotate_image(gray_image, heading)
        #1920 x 972 = 1080 * 0.9 = 972
        # lets normalize first based on stored headings 
        temp_time_1 = time.time()
        keypoints, descriptors = self.global_detector.get_keydes(gray_image)
        temp_time_2 = time.time() - temp_time_1
        self.keypoint_time += temp_time_2
        self.keypoint_iterations += 1
        if len(keypoints) < 10:
            print(f"WarningA: Not enough keypoints found for image {index}. Skipping.")

        self.stored_global_keypoints.append(keypoints)
        self.stored_global_descriptors.append(descriptors)

        # LOCAL PHASE
        kp1, des1 = None, None
                
        if (self.global_detector_name == self.local_detector_name and self.same_detector_threshold == True) and self.neural_net_on == False:
            kp1, des1 = keypoints, descriptors
            if len(kp1) < 10:
                print(f"WarningB: Not enough keypoints found for image {index}. Decrease keypoint threshold - cond 1)")
            self.stored_local_keypoints.append(kp1)
            self.stored_local_descriptors.append(des1)


        elif (self.global_detector_name != self.local_detector_name or self.same_detector_threshold == False) and self.neural_net_on == False:
            kp1, des1 = self.local_detector.get_keydes(gray_image)
            self.stored_local_keypoints.append(kp1)
            self.stored_local_descriptors.append(des1)
            if len(kp1) < 10:
                print(f"WarningB: Not enough keypoints found for image {index}. Decrease keypoint threshold - cond 2)")


        elif self.neural_net_on == True:
                # extract features with Superpoint
                features = self.local_detector.get_features(gray_image)
                self.stored_feats.append(features)

    


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
    
    def store_and_append_stability(self, act_x, act_y, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, tx5, ty5):
        # method_1 = "Phase_Correlate"
        # method_2 = "LinAlgPts"
        # method_5 = "Affine"
        # method_6 = "Rigid"
        # method_7 = "Homography"
        magactual = np.linalg.norm([act_x, act_y])
        magnitude1 = np.linalg.norm([tx1, ty1])
        magnitude2 = np.linalg.norm([tx2, ty2])
        magnitude3 = np.linalg.norm([tx3, ty3])
        magnitude4 = np.linalg.norm([tx4, ty4])
        magnitude5 = np.linalg.norm([tx5, ty5])

        mag1_stability = magnitude1 / magactual
        mag2_stability = magnitude2 / magactual
        mag3_stability = magnitude3 / magactual
        mag4_stability = magnitude4 / magactual
        mag5_stability = magnitude5 / magactual
        self.mag1_stability_arr.append(mag1_stability)
        self.mag2_stability_arr.append(mag2_stability)
        self.mag3_stability_arr.append(mag3_stability)
        self.mag4_stability_arr.append(mag4_stability)
        self.mag5_stability_arr.append(mag5_stability)
        



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



    def translate_image(self, image, shift_x, shift_y):
        """Translate the image by a given x and y shift."""
        height, width = image.shape[:2]
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
        return translated_image
    
    def add_no_GPS_image(self, index, directory):
        gray_image = self.pull_image(index, directory, bool_rotate=True, use_estimate=True)
        #1920 x 972 = 1080 * 0.9 = 972
        # lets normalize first based on stored headings 
        keypoints, descriptors = self.global_detector.get_keydes(gray_image)
        if len(keypoints) < 100:
            print(f"WarningA: Not enough keypoints found for image {index}. cond - A.")



        if index >= len(self.global_loss_keypoints):
            self.global_loss_keypoints.extend([None] * (index + 1 - len(self.global_loss_keypoints)))
        self.global_loss_keypoints[index] = keypoints

        if index >= len(self.global_loss_descriptors):
            self.global_loss_descriptors.extend([None] * (index + 1 - len(self.global_loss_descriptors)))
        self.global_loss_descriptors[index] = descriptors


        kp1, des1 = None, None
        if (self.global_detector_name != self.local_detector_name) and self.neural_net_on == False:
            kp1, des1 = self.local_detector.get_keydes(gray_image)
            if index >= len(self.local_loss_keypoints):
                self.local_loss_keypoints.extend([None] * (index + 1 - len(self.local_loss_keypoints)))
            self.local_loss_keypoints[index] = kp1
            if index >= len(self.local_loss_descriptors):
                self.local_loss_descriptors.extend([None] * (index + 1 - len(self.local_loss_descriptors)))
            self.local_loss_descriptors[index] = des1
            if len(kp1) < 100:
                print(f"WarningB: Kp under 100 for image {index}. Decrease keypoint threshold - cond 3)")            

        elif (self.global_detector_name == self.local_detector_name and self.same_detector_threshold == True) and self.neural_net_on == False:
            kp1, des1 = self.global_loss_keypoints[index], self.global_loss_descriptors[index]
            if index >= len(self.local_loss_keypoints):
                self.local_loss_keypoints.extend([None] * (index + 1 - len(self.local_loss_keypoints)))
            self.local_loss_keypoints[index] = self.global_loss_keypoints[index]

            if index >= len(self.local_loss_descriptors):
                self.local_loss_descriptors.extend([None] * (index + 1 - len(self.local_loss_descriptors)))
            self.local_loss_descriptors[index] = self.global_loss_descriptors[index]
            if len(kp1) < 100:
                print(f"WarningB: Kp under 100 for image {index}. Decrease keypoint threshold - cond 4)")
        
        elif  (self.global_detector_name == self.local_detector_name and self.same_detector_threshold != True) and self.neural_net_on == False:
            kp1, des1 = self.local_detector.get_keydes(gray_image)
            if index >= len(self.local_loss_keypoints):
                self.local_loss_keypoints.extend([None] * (index + 1 - len(self.local_loss_keypoints)))
            self.local_loss_keypoints[index] = kp1
            if index >= len(self.local_loss_descriptors):
                self.local_loss_descriptors.extend([None] * (index + 1 - len(self.local_loss_descriptors)))
            self.local_loss_descriptors[index] = des1
            if len(kp1) < 100:
                print(f"WarningB: Kp under 100 for image {index}. Decrease keypoint threshold - cond 5)")




        if self.neural_net_on == True:
                # extract features with Superpoint
                features = self.local_detector.get_features(gray_image)
                if index >= len(self.local_loss_feats):
                    self.local_loss_feats.extend([None] * (index + 1 - len(self.local_loss_feats)))
                self.local_loss_feats[index] = features



    def append_answers_estimates(self, mean_pixel_changes_x, mean_pixel_changes_y, actual_pixel_change_x_m, actual_pixel_change_y_m):
        self.estimated_gps_x_inference.append(mean_pixel_changes_x)
        self.estimated_gps_y_inference.append(mean_pixel_changes_y)
        self.actuals_x.append(actual_pixel_change_x_m)
        self.actuals_y.append(actual_pixel_change_y_m)
    
    def print_stability_analysis(self):
        print("Stability analysis mean (relative to src - best result) and var*10e6")
        print(f"Phase Correlation stability: {np.mean(self.mag1_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag1_stability_arr, ddof=1)*1000000):.3f}")
        print(f"Linear Algebra stability: {np.mean(self.mag2_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag2_stability_arr, ddof=1)*1000000):.3f}")
        print(f"Affine stability: {np.mean(self.mag3_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag3_stability_arr, ddof=1)*1000000):.3f}")
        print(f"Rigid stability: {np.mean(self.mag4_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag4_stability_arr, ddof=1)*1000000):.3f}")
        print(f"Homography stability: {np.mean(self.mag5_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag5_stability_arr, ddof=1)*1000000):.3f}")
        print("")
        string_stability = f"Stability analysis mean (relative to src - best result) and var*10e6\nPhase Correlation: {np.mean(self.mag1_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag1_stability_arr, ddof=1)*1000000):.3f}\nSVD: {np.mean(self.mag2_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag2_stability_arr, ddof=1)*1000000):.3f}\nAffine Transform: {np.mean(self.mag3_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag3_stability_arr, ddof=1)*1000000):.3f}\nRigid Transform: {np.mean(self.mag4_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag4_stability_arr, ddof=1)*1000000):.3f}\nHomography Transform: {np.mean(self.mag5_stability_arr)/np.mean(self.mag2_stability_arr):.3f} +/- {(np.var(self.mag5_stability_arr, ddof=1)*1000000):.3f}\n"
        self.string_stability = string_stability



    def analyze_matches(self, bool_infer_factor, num_images_analyze):
        deviation_norms_x = []
        deviation_norms_y = []

        range_im = num_images_analyze

        for i in reversed(range(1, range_im)):  # iterate through all images in reverse order
            
            best_index = -1   
            internal_angle = None
            image_space = self.find_image_space(i)
            timeAS = time.time()
            match_time_1 = time.time()
            if self.global_matching_technique < 3:
                best_index, internal_angle = self.find_best_match(i, image_space)           
              
            else: 
                best_index, internal_angle = self.find_best_match_multimethod(i, image_space)
            match_time_2 = time.time() - match_time_1
            
            if best_index != -1 and internal_angle is not None:
                time_rand = time.time()
                if not bool_infer_factor:
                    self.add_no_GPS_image(i, self.directory)
                # please note: the images in this dataset have assumed headings based on testing, as such they are not exact, and images that were meant to have perfect reference headings have estimated ones, thus adding a partial error to the system. 

                # actual GPS data to compare against
                actual_gps_diff_meters = (np.array(self.stored_gps[i]) - np.array(self.stored_gps[best_index])) * 111139
                actual_pixel_change_x_m, actual_pixel_change_y_m = actual_gps_diff_meters[0], actual_gps_diff_meters[1]
                self.actual_GPS_deviation.append((np.abs(actual_pixel_change_x_m), np.abs(actual_pixel_change_y_m)))
                Lowes_ratio = 0.8 if self.local_detector_name == "AKAZE" else 0.7
                if bool_infer_factor:
                    # print(f"len feats: {len(self.stored_feats[i]['keypoints'])}") if self.neural_net_on == True 
                    src_pts, dst_pts = self.get_src_dst_pts(self.stored_local_keypoints[i], self.stored_local_keypoints[best_index], self.stored_local_descriptors[i], self.stored_local_descriptors[best_index], Lowes_ratio, global_matcher_true=False) if self.neural_net_on == False else get_neural_src_pts(self.stored_feats[i], self.stored_feats[best_index])
                elif not bool_infer_factor:
                    src_pts, dst_pts = self.get_src_dst_pts(self.local_loss_keypoints[i], self.stored_local_keypoints[best_index], self.local_loss_descriptors[i], self.stored_local_descriptors[best_index], Lowes_ratio, global_matcher_true=False) if self.neural_net_on == False else get_neural_src_pts(self.local_loss_feats[i], self.stored_feats[best_index])
                    len_matches = len(src_pts)
                    self.len_matches_arr.append(len_matches)
                
                print(f"len matches: {len(src_pts)}")    

                if len(src_pts) < 100 or len(dst_pts) < 100:
                    print("Low number of keypoints found.")
                time_rand_2 = time.time() - time_rand
                
                shift_time_1 = time.time()
                translation_x, translation_y = 0, 0
                # if self.neural_net_on == True:
                #     translation_x, translation_y = get_shifts(self.local_loss_feats[i], self.stored_feats[best_index]) 
                # else:  
                translation_method = 0
                tx1, ty1, tx2, ty2, tx3, ty3, deg3, tx4, ty4, deg4, tx5, ty5, deg5 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                if 1==1: #translation_method == 0:
                    # phase corr
                    temp_err_est_act_heading = np.abs(self.stored_headings[i]) - np.abs(self.estimated_headings[i])
                    if temp_err_est_act_heading > 2.5:
                        print("Heading estimation error too high.")
                    image_to_infer_normed = None
                    if bool_infer_factor:
                        image_to_infer_normed = self.pull_image(i, self.directory, bool_rotate=True, use_estimate=False) # comparison with reference images
                    else:
                        image_to_infer_normed = self.pull_image(i, self.directory, bool_rotate=True, use_estimate=True) # comparison between ref and inference
                    reference_image_normed = self.pull_image(best_index, self.directory, bool_rotate=True) # ref 

                    # image_to_infer_normed = cv2.GaussianBlur(image_to_infer_normed, (3, 3), 0)
                    # reference_image_normed = cv2.GaussianBlur(reference_image_normed, (3, 3), 0) # global matching is highly noise variant
                    shift, _ = cv2.phaseCorrelate(np.float32(image_to_infer_normed), np.float32(reference_image_normed))
                    
                    tx1, ty1 = shift
                    # translation_x, translation_y = tx1, ty1

                #elif translation_method == 1:
                    # direct src normalization and computation
                    tx2, ty2 = get_src_shifts(src_pts, dst_pts)
                    
                                                        
                #elif translation_method == 2:
                    src_pts = src_pts.reshape(-1,2)
                    dst_pts = dst_pts.reshape(-1,2)
                    # affine: translation, rotation, scale, shear
                    tx3, ty3, deg3 = affine_transformation(src_pts, dst_pts)
                    # translation_x, translation_y = tx5, ty5 
                #elif translation_method == 3:
                    # rigid: translation, rotation (manual method)
                    # src_pts = src_pts.reshape(-1,2)
                    # dst_pts = dst_pts.reshape(-1,2)
                    tx4, ty4, deg4 = rigid_transformation(src_pts, dst_pts)
                    # translation_x, translation_y = tx6, ty6
                #elif translation_method == 4:
                    # homography: translation, rotation, scale, shear, perspective
                    src_pts = src_pts.reshape(-1,2)
                    dst_pts = dst_pts.reshape(-1,2)
                    tx5, ty5, deg5 = homography_transformation(src_pts, dst_pts)
                    # translation_x, translation_y = tx7, ty7 
            
                    translation_x, translation_y = tx2, ty2
                if not bool_infer_factor:
                    self.store_and_append_stability(actual_pixel_change_x_m, actual_pixel_change_y_m, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, tx5, ty5)    

                shift_time_2 = time.time() - shift_time_1
                
             


                # INFERENCE
                if bool_infer_factor and actual_pixel_change_x_m != 0 and actual_pixel_change_y_m != 0:
                    self.append_answers_estimates(translation_x, translation_y, actual_pixel_change_x_m, actual_pixel_change_y_m)
    
                # pixel -> meters
                translation_x_m = translation_x * self.inferred_factor_x 
                translation_y_m = translation_y * self.inferred_factor_y

                # NEW GPS
                new_lon = self.stored_gps[best_index][0] + translation_x_m / 111139
                new_lat = self.stored_gps[best_index][1] + translation_y_m / 111139
                if not bool_infer_factor:
                    self.estimated_gps.append((new_lon, new_lat))


                # DEBUG
                deviation_x_meters = translation_x_m - actual_pixel_change_x_m
                deviation_y_meters = translation_y_m - actual_pixel_change_y_m
                deviation_norms_x.append((deviation_x_meters))
                deviation_norms_y.append(np.abs(deviation_y_meters))
                # print(f"DEV-X,Y (m): {deviation_x_meters}, {deviation_y_meters}  for im {i+1} wrt {best_index+1}, angle: {((self.stored_headings[i])-(self.estimated_headings[i])):.4f} deg, actual deviation (m): {actual_pixel_change_x_m}, {actual_pixel_change_y_m}")


                time_BB = time.time() - timeAS
                # print("Match time: ", match_time_2, "Shift time: ", shift_time_2, "Total time: ", time_BB)

       
        mean_x_dev, mean_y_dev = np.mean(deviation_norms_x), np.mean(deviation_norms_y)
        if not bool_infer_factor:
            self.print_stability_analysis()
        


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


def display_side_by_side(image1, image2, mean_x, mean_y):
    # Ensure both images are loaded
    if image1 is None or image2 is None:
        print("Error: Could not load one or both images.")
        return

    # Resize images to a quarter of their original size
    height1, width1 = image1.shape[:2]
    quarter_size_image1 = cv2.resize(image1, (width1 // 4, height1 // 4))
    quarter_size_image2 = cv2.resize(image2, (width1 // 4, height1 // 4))

    # Invert colors for visibility (negative image effect)
    inverted_image1 = cv2.bitwise_not(quarter_size_image1)
    inverted_image2 = cv2.bitwise_not(quarter_size_image2)

    # Get dimensions of the resized image1
    height1, width1 = inverted_image1.shape[:2]

    # Draw a line on the first image from the center to the point + mean_x, mean_y
    center = (width1 // 2, height1 // 2)
    end_point = (int(center[0] + mean_x / 4), int(center[1] + mean_y / 4))  # Adjusting for resized image
    line_image1 = inverted_image1.copy()
    cv2.line(line_image1, center, end_point, (0, 255, 0), thickness=2)  # Green line

    # Stack the images horizontally
    combined_image = np.hstack((line_image1, inverted_image2))

    # Display the result
    cv2.imshow("Images Side by Side with Line on Image 1", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_image(img):
    """Crop the top and bottom 10% of the image."""

    height, width = img.shape[:2]
    crop_size = int(height * 0.05)  # 10% of the height
    # cropped_image = cv2.GaussianBlur(cropped_image, (kernel_to_test, kernel_to_test), 0)  # Denoise
    cropped_image = img[crop_size:height-crop_size, :]
    # overlay_images(cropped_image, cropped_image)
    return img[crop_size:height-crop_size, :]  # Crop top and bottom




def normalize_images_to_global_coord_system(ref_img, inference_image, reference_heading, internal_angle):
    rotation_angle = -internal_angle#-estimate_affine_rotation(inference_image, ref_img)
    rot_inf_img = rotate_image(inference_image, rotation_angle + reference_heading)
    rot_ref_img = rotate_image(ref_img, reference_heading)
    return rot_inf_img, rot_ref_img




def normalize_translation_to_global_coord_system(tx, ty, reference_heading, internal_angle):
    # The same rotation angle logic as the original function
    theta_inf = reference_heading - internal_angle
    
    # Rotation matrix for src_pts (inference points) - rotate by (rotation_angle + reference_heading)
    rotation_matrix_src = np.array([[np.cos(np.radians(theta_inf)), -np.sin(np.radians(theta_inf))], [np.sin(np.radians(theta_inf)),  np.cos(np.radians(theta_inf))]])
    # print(f"rot angle: {rotation_angle}, ref: {reference_heading}")
    # Rotate translation using the first rotation matrix
    rot_translation = np.dot(np.array([tx, ty]), rotation_matrix_src.T)

    return rot_translation[0], rot_translation[1]



def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0
    if dir in ['S', 'W']:
        decimal = -decimal
    return decimal




def rotate_image(image, angle):
    """Rotate the image by a given angle around its center."""
    if angle is None:
        print(f"Nothing done")
        return image
    

    height, width = image.shape[:2]
    # print(f"h, w: {height}, {width}")
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image




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
    elif local_detector_choice == 3:
        printable_loc_detector = "SUPERPOINT"

    if local_detector_choice == 3:
        printable_loc_matcher = "LightGlue"
    elif local_matcher_choice == 0:
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

    


# ASD
    # Settings (curr_best: akaze,bf, affine, histo, akaze,flann)
    global_detector_choice = 2  # # Set 1 for ORB, 2 for AKAZE
    global_matcher_choice = 0  # Set 0 for BFMatcher, 1 for FlannMatcher, 2 for GraphMatcher
    global_matcher_technique = 4  # Set 0 for Pass Through grid counter, 3 for correlation, 4 for histogram, 5 for SSIM, 6 for Hash (DONT USE HASHING). 
    local_detector_choice = 1  # Set 1 for ORB, 2 for AKAZE 3 for NEURAL (lightglue matcher)
    local_matcher_choice = 0  # Set 0 for BFMatcher, 1 for FlannMatcher, 2 for GraphMatcher


    # DATASET optimization
    main_dataset_name = "DATSETAMAZ" 

    # AKAZE thresh, ORB kp. 
    num_images = 15
    inference_images = 7

    global_detector_arr = [1,2]
    global_matcher_arr = [0,1,2]
    global_matcher_technique_arr = [0,3,4,5]
    local_detector_arr = [1,2, 3]
    local_matcher_arr = [0,1,2]
    iteration_count = 0
    directory = './GoogleEarth/DATASETS/{}'.format(main_dataset_name)
    if 1==1:
    
        # for global_matcher_choice in global_matcher_arr:
    #         for global_matcher_technique in global_matcher_technique_arr:
                # for local_detector_choice in local_detector_arr:
                #     for global_detector_choice in global_detector_arr:
    #                 for local_matcher_choice in local_matcher_arr:
    #                     # We only want to run one instance of neural mode, others are redundant.
    #                     if local_detector_choice == 3 and local_matcher_choice !=0:
    #                         continue
                        # if the navigator object exists
                        if 'navigator' in locals():
                            # navigator.reset_all_data()
                            del navigator
                            gc.collect()
                            print(f'End of Iteration: {iteration_count}') 
                            
                        if main_dataset_name == "DATSETROT" or main_dataset_name == "DATSETCPT": 
                            glob_thresh=0.00352 if global_detector_choice == 2 else 3000 if global_detector_choice == 1 else 0
                        elif main_dataset_name == "DATSETROCK":
                            glob_thresh=0.001  if global_detector_choice == 2 else 3000 if global_detector_choice == 1 else 0
                        loc_det_thresh = 0.001 if local_detector_choice == 2 else 3000 if local_detector_choice == 1 else 0

                        if main_dataset_name == "DATSETROT":
                            glob_thresh = 0.000017 if global_detector_choice == 2 else 20000 if global_detector_choice == 1 else 0
                            loc_det_thresh = 0.00001 if local_detector_choice == 2 else 20000 if local_detector_choice == 1 else 0
                        elif main_dataset_name == "DATSETAMAZ":
                            glob_thresh = 0.00027 if global_detector_choice == 2 else 20000 if global_detector_choice == 1 else 0
                            loc_det_thresh = 0.00007 if local_detector_choice == 2 else 20000 if local_detector_choice == 1 else 0




                        main_start_time = time.time()
                        navigator = UAVNavigator(global_detector_choice, local_detector_choice , global_matcher_choice, local_matcher_choice, global_matcher_technique, main_dataset_name, glob_thresh, loc_det_thresh) # INITIALIZATION
                        init_time = time.time() - main_start_time
                        # _ = print_choices(global_detector_choice,global_matcher_choice, global_matcher_technique, local_detector_choice, local_matcher_choice) XXX put back

                        iteration_count += 1
                        
                        # navigator.clear_stored_data()  # Clear stored data before each kernel test

                        # Step 1: Add images and infer factors
                        
                        for i in range(1, inference_images + 1): # stops at inference_images = 6]
                            navigator.add_image(i, directory)

                        # analyse Deviations to get inferred factors. BOOL_INFER_FACTOR = TRUE
                        navigator.analyze_matches(True, inference_images)
                        navigator.compute_linear_regression_factors()
                        
                        for i in range(inference_images+1, num_images + 1):
                            navigator.add_image(i, directory)
                        
                        # BOOL INFER FACTOR = FALSE. num_images_analyze = 13
                        navigator.analyze_matches(False, num_images)


                        #DEBUG
                        #conv np arr
                        np_act_change = np.array(navigator.actual_GPS_deviation)
                        np_est_dev = np.array(navigator.norm_GPS_error)
                        # find radial mean movement of UAV
                        gps_act_x, gps_act_y = np.mean(np_act_change[:,0]), np.mean(np_act_change[:,1])
                        mean_radial_movement = np.sqrt(gps_act_x**2 + gps_act_y**2)
                        # find swing percentage
                        swing_percent = np.array(100*np_est_dev/mean_radial_movement)
                        print(f"Percentage Deviation: {swing_percent} %")
                        string_percent_GPS_dev = f"Percentage GPS Deviation: {swing_percent} %"
                        string_params = print_choices(global_detector_choice,global_matcher_choice, global_matcher_technique, local_detector_choice, local_matcher_choice)
                        string_GPS_error = f"Mean normalized GPS error: {np_est_dev}"
                        string_heading_error = f"Mean Heading Error: {np.mean(navigator.estimated_heading_deviations)}"
                        print(string_GPS_error, '\n', string_heading_error)
                        




                        num_global_keypoints_per_image = [len(kps) for kps in navigator.stored_global_keypoints] # XXX sort out
                                              
                        string_mean_glob_len_kp = f"Mean Length of Global Keypoints: {np.mean(num_global_keypoints_per_image)}"
                        print(string_mean_glob_len_kp) # XXX append this later to file
                        
                        num_local_keypoints_per_image = [len(kps) for kps in navigator.stored_local_keypoints] if local_detector_choice != 3 else [len(kps) for kps in navigator.stored_feats[0]['keypoints']]
                        string_mean_len_local_kp = f"Mean Length of Local Keypoints: {np.mean(num_local_keypoints_per_image)}"
                        print(string_mean_len_local_kp) # XXX append this later to file
                        max_min_loc_range_kp = np.max(num_local_keypoints_per_image) - np.min(num_local_keypoints_per_image)
                        string_max_min_range_matches = f"Range of Local kp: {max_min_loc_range_kp}"
                        print(string_max_min_range_matches) # XXX append this later to file
                        mean_loc_matches = np.mean(navigator.len_matches_arr) 
                        
                        string_mean_time_kp = f"Mean Global Time to Extract Keypoints: {navigator.keypoint_time/navigator.keypoint_iterations:.4f} s"
                        print(string_mean_time_kp) # XXX append this later to file
                        mean_good_matches = np.mean(navigator.glob_mat_len_arr)
                        string_mean_good_matches = f"Mean Number of Glob good Matches: {mean_good_matches}"
                        print(string_mean_good_matches) # XXX append this later to file


                        append_to_file("results.txt", string_params, string_GPS_error, string_percent_GPS_dev, string_heading_error, navigator.string_stability, "\n\n")

                        # print the mean, max and min of the navigator, runtime rotational normalizer runtime_rotational_estimator
                        # print(f"Mean Rotational Normalizer Time: {np.mean(navigator.runtime_rotational_estimator)} , Max Rotational Normalizer Time: {np.max(navigator.runtime_rotational_estimator)}, Min Rotational Normalizer Time: {np.min(navigator.runtime_rotational_estimator)} ms, median: {np.median(navigator.runtime_rotational_estimator)}") if 


                        elapsed_time = time.time() - main_start_time 
                        print(f"Time taken to execute The Method: {elapsed_time:.4f} seconds")
                        
                        
                        # print the mean, max and min of the navigator, runtime_rotational_estimator        
                    
                    
    # ANY code outside of all iterations. 
    


if __name__ == "__main__":
    main() 
