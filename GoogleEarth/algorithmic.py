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
    def __init__(self, global_detector_choice, local_detector_choice, rotational_detector_choice, global_matcher_choice, local_matcher_choice, global_matching_technique, dataset_name, rotation_method, global_detector_threshold=0.001, local_detector_threshold=0.001, rotational_detector_threshold=0.001, translation_method=0, scale_factor=1):
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

        

        self.scale_factor_dual = 0
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
        self.MAE_GPS = []
        

        # Neural Network
        self.stored_feats = []


        # coverage
        self.x_overlap = []
        self.y_overlap = []
    
        # Translation
        self.translation_method = translation_method


        # Stability of translation estimations DEBUG
        self.mag1_stability_arr = []
        self.mag2_stability_arr = []
        self.mag3_stability_arr = []
        self.mag4_stability_arr = []
        self.mag5_stability_arr = []
        self.string_stability = []



        self.same_detector_threshold = False

        # DEBUG 
        self.keypoint_time = 0 
        self.keypoint_iterations = 0
        self.len_matches_arr = []
        self.glob_mat_len_arr = []
        self.loc_mat_len_arr = []
        self.location_inference_time_arr = []
        self.parameter_inference_time_arr = []
        self.add_time_arr = []



        if local_detector_choice == 3:
            self.neural_net_on = True

        if global_detector_threshold == local_detector_threshold:
            self.same_detector_threshold = True
        
        # datasets
        self.dataset_name = dataset_name # pass as string
        self.directory = './GoogleEarth/DATASETS/{}'.format(self.dataset_name)

        # rotation
        self.method_to_use = rotation_method # 0 is affine, 1 is homography, 2 is partial affine

        # timing 
        self.time_best_match_function = 0
        self.runtime_rotational_estimator = []
        efficiency_factor = 2/3
        if global_detector_choice == 1:
            self.global_detector_name = "ORB"
            self.global_detector = set_feature_extractor(global_detector_choice, global_detector_threshold)
        elif global_detector_choice == 2: 
            
            self.global_detector_name = "AKAZE"
            self.global_detector = set_feature_extractor(global_detector_choice, global_detector_threshold)
            redo = self.test_and_reinitialize(self.global_detector, 2000*efficiency_factor)
            while redo:
                global_detector_threshold *= 0.75
                self.global_detector = set_feature_extractor(global_detector_choice, global_detector_threshold)
                redo = self.test_and_reinitialize(self.global_detector, 2000*efficiency_factor)
            
        if rotational_detector_choice == 1:                
            self.rotational_detector_name = "ORB"
            self.rotational_detector = set_feature_extractor(rotational_detector_choice, rotational_detector_threshold)
        elif rotational_detector_choice == 2: 
            self.rotational_detector_name = "AKAZE"
            self.rotational_detector = set_feature_extractor(rotational_detector_choice, rotational_detector_threshold)
            redo = self.test_and_reinitialize(self.rotational_detector, 3000*efficiency_factor)
            countA = 0
            while redo:
                countA += 1
                rotational_detector_threshold *= 0.5
                self.rotational_detector = set_feature_extractor(rotational_detector_choice, rotational_detector_threshold)
                redo = self.test_and_reinitialize(self.rotational_detector, 3000*efficiency_factor)
                if countA > 13:
                    break
            

        if local_detector_choice == 1:                
            self.local_detector_name = "ORB"
            self.local_detector = set_feature_extractor(local_detector_choice, local_detector_threshold)
        elif local_detector_choice == 2: 
            self.local_detector_name = "AKAZE"
            self.local_detector = set_feature_extractor(local_detector_choice, local_detector_threshold)
            redo = self.test_and_reinitialize(self.local_detector, 3000*efficiency_factor)
            countA = 0
            while redo:
                countA += 1
                local_detector_threshold *= 0.5
                self.local_detector = set_feature_extractor(local_detector_choice, local_detector_threshold)
                redo = self.test_and_reinitialize(self.local_detector, 3000*efficiency_factor)
                if countA > 10:
                    break
            

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


    


    def test_and_reinitialize(self, detector, num_kps=1000):
        # find kps, if not enough, reinitialize
        # image_path = os.path.join(self.directory, f'{1}.jpg')
        # image = cv2.imread(image_path)  # Read the image in color
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_image = self.pull_image(1, self.directory)
        kps, _ = detector.get_keydes(gray_image)
        redo_true = True if len(kps) < num_kps else False
        return redo_true


    def get_rotations(self, src_pts, dst_pts, method_to_use=2):
        homography_threshold = 25 if self.global_detector_name == "ORB" else 0.5

        M = None
        mask = None
        if method_to_use == 0:
            # This needs to actually be tuned. 
            M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)



            
        elif method_to_use == 1:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
            M = M / M[2, 2]
        
            # The top-left 2x2 submatrix contains the rotational and scaling info
            rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)
            scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)

        elif method_to_use == 2:
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

        if M is None:
            raise ValueError("Transformation matrix estimation failed.")

        return M, np.sum(mask)
    



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



            
            if self.local_detector_name == "ORB" and self.dataset_name == "DATSETSAND":
                Lowes_thresh -= 0.25
            else:
                Lowes_thresh -= 0.2 # XXX

            good_matches = []
            count = 0
            while len(good_matches) < 500:
                count += 1
                good_matches = []
                for match_pair in matches:
                    # BFMatcher and FlannMatcher return pairs (tuple of two matches)
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < Lowes_thresh * n.distance:
                            good_matches.append(m)
                    elif len(match_pair) == 1:
                        pass
                if count > 30:
                    break
                Lowes_thresh += 0.025 
            # sort good matches according to distance
          
                        
            if global_matcher_true:
                self.glob_mat_len_arr.append(len(good_matches)) 
            elif not global_matcher_true:
                self.loc_mat_len_arr.append(len(good_matches))
            # Extract matched keypoints 


            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            
            return src_pts, dst_pts, good_matches

    
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
        

        # print(f"len of image space: {len(image_space)}")
        for ref in image_space:
            
            if ref == image_index:  # Can change this to if i >= image_index to only infer from lower indices
                continue  
            ref_img = self.pull_image(ref, self.directory, bool_rotate=False)
            kp_stored = self.stored_global_keypoints[ref]
            descriptors_stored = self.stored_global_descriptors[ref]
            Lowes_ratio = 0.8 if self.global_detector_name == "AKAZE" else 0.7
            if self.local_matcher_choice == 0:
                Lowes_ratio +=0.1
            src_pts, dst_pts, _ = self.get_src_dst_pts(kp_inf, kp_stored, des_inf, descriptors_stored, Lowes_ratio)

            method_to_use = self.method_to_use # 0 is affine, 1 is homography
            M, amt_inliers = self.get_rotations(src_pts, dst_pts, method_to_use) # inf = src = img_index, dst = ref = i = stored
            # print(f"AMT INLIERS: {amt_inliers}\n")
            
            if amt_inliers < 80 or M is None:
                continue
            

            h, w = ref_img.shape[:2]
            if method_to_use == 0:
                rot_ref = cv2.warpAffine(ref_img, M[:2, :], (w, h))
            elif method_to_use == 1:
                rot_ref = cv2.warpPerspective(ref_img, M, (w, h))
            elif method_to_use == 2:
                rot_ref = cv2.warpAffine(ref_img, M, (w, h))

            # for robustness test, lets impose a 1 degree rotation offset onto the reference image, only experienced by the global matching technique
            # lets first find a gaussian angle between -10 and 10 degrees
            
            
            # Perform cross-correlation directly on the full image
            if self.global_matching_technique == 3:
                score = cross_correlate(inference_img, rot_ref)


            elif self.global_matching_technique == 4:
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
                


        # heading_change = 0
        # if len(linked_angles) > 0:
        #     heading_change = -linked_angles[-1]
            # self.compare_headings(image_index, best_index, heading_change)
        
        return best_index, -0

    def pull_image(self, index, directory, bool_rotate=False, use_estimate=False):
        """Pull an image from from the directory with name index.jpg"""
        image_path = os.path.join(directory, f'{index+1}.jpg')

        image = cv2.imread(image_path)  # Read the image in
        cropped_image = self.crop_image(image)

#         # # filter
        alpha = 0.5  # Contrast control (0.0-1.0, where <1 reduces contrast)
        beta = -50   # Brightness control (negative for darkening)
        low_light_image = cv2.convertScaleAbs(cropped_image, alpha=alpha, beta=beta)
        noise_sigma = 25
        noise = np.random.normal(0, noise_sigma, cropped_image.shape).astype(np.uint8)
        low_light_noisy_image = cv2.addWeighted(low_light_image, 0.9, noise, 0.1, 0)
        tinted_image = cv2.addWeighted(low_light_noisy_image, 0.7, np.zeros_like(cropped_image), 0.3, 0)
        
        # cv2.imshow("Tinted Low Light Image", tinted_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cropped_image = tinted_image
        
            # Convert to grayscale for detectors if not already
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image

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
        # """Compares two images using average hash."""
        # img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        # img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        # hash1 = imagehash.average_hash(img1_pil)
        # hash2 = imagehash.average_hash(img2_pil)
        # score = 1 - (hash1 - hash2) / len(hash1.hash)  # Normalized Hamming distance
        # lets rather do phase correlation and take confidence as the score.
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        _, response = cv2.phaseCorrelate(img1, img2)
        score = response*1000

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
        
        # if inference_index >= len(self.estimated_headings):
        #         # Extend the list with None or a default value up to the required index
        #         self.estimated_headings.extend([None] * (inference_index + 1 - len(self.estimated_headings)))
        # self.estimated_headings[inference_index] = estimated_new_heading
        
        # self.estimated_heading_deviations.append(np.abs(deviation_heading))

    def find_good_matches(self, matches, Lowe_thresh):
        good_matches = []
        for match_pair in matches:
                # BFMatcher and FlannMatcher return pairs (tuple of two matches)
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < Lowe_thresh * n.distance:  # Lowe's ratio test, for rotations
                        good_matches.append(m)

        return good_matches



    def find_best_match(self, image_index, image_space, grid_size=(4, 4)):
        crude_detector = set_feature_extractor(1, 200) # detector choice : 1 is ORB, 1500 kp
        crude_matcher = set_matcher("flann_matcher")


        def divide_into_grids(image, grid_size):
            """Divides the given image into grid_size (rows, cols) and returns the grid segments."""
            height, width = image.shape[:2]
            grid_height = height // grid_size[0]
            grid_width = width // grid_size[1]
            grids = []

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    # Handle the last row/column differently to account for any remaining pixels
                    grid = image[i*grid_height:(i+1)*grid_height if i != grid_size[0]-1 else height,
                                j*grid_width:(j+1)*grid_width if j != grid_size[1]-1 else width]

                    # Check if grid is valid and not empty
                    if grid.size == 0:
                        print(f"Empty grid at row {i}, col {j}")
                        continue
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
            
            ref_img = self.pull_image(ref, self.directory, bool_rotate=False)
            kp_stored = self.stored_global_keypoints[ref]
            descriptors_stored = self.stored_global_descriptors[ref]

            detector_choice = 1 if self.global_detector_name == "ORB" else 2
            Lowes_Ratio = 0.8 if self.global_detector_name == "AKAZE" else 0.7 
            if self.local_matcher_choice == 0:
                Lowes_Ratio +=0.1
            
            src_pts, dst_pts, _ = self.get_src_dst_pts(kp_inf, kp_stored, des_inf, descriptors_stored, Lowes_Ratio)
 
            # initial rotating of image. 
            if len(src_pts) > 10:
                # Extract matched points

                method_to_use = self.method_to_use # 0 is affine, 1 is homography # 2 is partial affine
                M, amt_inliers = self.get_rotations(src_pts, dst_pts, method_to_use) 

                if amt_inliers < 80 or M is None:
                    continue

                # Rotate the stored image using the homography matrix
                h, w = ref_img.shape[:2]
                if method_to_use == 0:
                    rotated_image_stored = cv2.warpAffine(ref_img, M[:2, :], (w, h))
                elif method_to_use == 1:
                    rotated_image_stored = cv2.warpAffine(ref_img, M,  (w, h))
                elif method_to_use == 2:
                    rotated_image_stored = cv2.warpAffine(ref_img, M, (w, h))
                

                # Now divide both current and rotated stored image into grids
                rotated_grids_stored = divide_into_grids(rotated_image_stored, grid_size)

                total_good_matches = 0

                # Perform grid-wise matching
                for current_grid, stored_grid in zip(current_grids, rotated_grids_stored):

                    if current_grid.shape[0] == 0 or stored_grid.shape[0] == 0 or current_grid.shape[1] == 0 or stored_grid.shape[1] == 0:
                        continue
                    if stored_grid.min() == stored_grid.max() == 0:
                        continue


                    kp_current_grid, descriptors_current_grid = crude_detector.get_keydes(current_grid)
                    kp_stored_grid, descriptors_stored_grid = crude_detector.get_keydes(stored_grid)


                    if descriptors_current_grid is None or descriptors_stored_grid is None or len(descriptors_current_grid) < 2 or len(descriptors_stored_grid) < 2:

                        continue

                    # Match descriptors between grids using knnMatch
                    detector_choice = 1 if self.global_detector_name == "ORB" else 2
                    matches_grid = crude_matcher.find_matches(descriptors_current_grid, descriptors_stored_grid, kp_current_grid, kp_stored_grid, detector_choice, global_matcher_true=1)
                    # check if matches grid is empty
                    if len(matches_grid) == 0:
                        continue

                    good_matches_grid = self.find_good_matches(matches_grid, 0.8)

                    total_good_matches += len(good_matches_grid) if len(good_matches_grid) < 20 else 20

                # Score based on total good matches across all grids
                score = total_good_matches

                if score > max_corr_score:
                    max_corr_score = score
                    best_index = ref
                    temp_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)

                    
                    linked_angles.append(temp_angle)


        heading_change = 0
        if len(linked_angles) > 0:
            heading_change = -linked_angles[-1]
            self.compare_headings(image_index, best_index, heading_change)

        return best_index, 0   #, -heading_change



    def find_image_space(self, image_index):
        

        # get the GPS of the current image
        current_gps = self.stored_gps[image_index]#self.estimated_gps[-1] if len(self.estimated_gps) > 0 else self.stored_gps[-1]
        # Get the most recent estimation of GPS, if available, otherwise use the stored GPS - the last value before GPS loss. 
        # get the GPS of all other images
        image_space = []
        prior_estimate = []
        iterative_radius = 1
        iterations = 0
        while len(image_space)<5:
            if iterations > 30:
                image_space = prior_estimate
                break
            image_space = []
            for i in range(len(self.stored_gps)):
                if i != image_index: # ensure, for Skripsie we dont infer from our own image.
                    stored_gps = self.stored_gps[i]
                    distance = np.linalg.norm(np.array(current_gps) - np.array(stored_gps)) # this 
                    

                    radial_distance_GPS = iterative_radius / 11139 #XXX
                    if distance < radial_distance_GPS:
                        image_space.append(i)
                        prior_estimate.append(i)
            if len(image_space) > 7:
                image_space = []
                iterative_radius-8
            else:
                
                iterative_radius += 100
            iterations += 1

        return image_space
     
    def find_closest_images(self, current_gps, image_index):
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371e3  # Earth radius in meters
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)

            a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            
            return R * c  # Distance in meters
        distances = []
        
        for i in range(len(self.stored_gps)):
            if i != image_index:  # Avoid the current image
                stored_gps = self.stored_gps[i]
                # Calculate Haversine distance
                distance = haversine(current_gps[0], current_gps[1], stored_gps[0], stored_gps[1])
                distances.append((i, distance))  # Store index and distance
        
        # Sort by distance (second item in the tuple)
        distances_sorted = sorted(distances, key=lambda x: x[1])

        # Select the top 5 closest images
        image_space = [i[0] for i in distances_sorted[:5]]  # Only keep indices of the closest images
        
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
        start_time_add = time.time()
        # image_path = os.path.join(directory, f'{index}.jpg')
        gps_path = os.path.join(directory, f'{index}.txt')

        
        # image = cv2.imread(image_path)  # Read the image in color
        gps_coordinates, heading = parse_gps(gps_path)
        """Add an image and its GPS coordinates to the stored list."""
        # cropped_image = self.crop_image(image)
        self.stored_gps.append(gps_coordinates)
        self.stored_headings.append(heading)
        self.stored_image_count += 1
        
        # # Convert to grayscale for detectors if not already
        # gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
        # # add im function is indexed with + 1
        gray_image = self.pull_image(index-1, directory)
        # gray_image = rotate_image(gray_image, heading)
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

        end_time_add = time.time() - start_time_add 
        self.add_time_arr.append(end_time_add)


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
    
    def append_answers_estimates(self, mean_pixel_changes_x, mean_pixel_changes_y, actual_pixel_change_x_m, actual_pixel_change_y_m):
        self.estimated_gps_x_inference.append(mean_pixel_changes_x)
        self.estimated_gps_y_inference.append(mean_pixel_changes_y)
        self.actuals_x.append(actual_pixel_change_x_m)
        self.actuals_y.append(actual_pixel_change_y_m)
    
    def reestimate_rotation(self, inference_index, best_index):
        """Reestimate the rotation angle based on the best match."""
        inference_img = self.pull_image(inference_index, self.directory, bool_rotate=False)
        # rotational_detector = set_feature_extractor(2, 200) # detector choice : 1 is ORB, 1500 kp. 2 is AKAZE, 1500 kp
        kp_inf, des_inf = self.rotational_detector.get_keydes(inference_img)
        ref_img = self.pull_image(best_index, self.directory, bool_rotate=False)
        ref_kp, ref_des = self.rotational_detector.get_keydes(ref_img)
        
        # kp_stored = self.stored_local_keypoints[best_index]
        # descriptors_stored = self.stored_local_descriptors[best_index]
        
        src_pts, dst_pts, _ = self.get_src_dst_pts(kp_inf, ref_kp, des_inf, ref_des, 0.8)

        M, inliers = self.get_rotations(src_pts, dst_pts, self.method_to_use)
        int_angle = np.arctan2(M[1, 0], M[0, 0]) * (180 / np.pi)


        return int_angle, inliers
    
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


    def lat_to_meters(self, lat_diff):
        # Earth's radius in meters
        R = 6371000  
        # Convert latitude difference to radians
        lat_diff_rad = np.radians(lat_diff)
        # Distance using Haversine formula for latitude difference
        return R * lat_diff_rad

    def lon_to_meters(self, lon_diff, reference_lat):
        # Earth's radius in meters
        R = 6371000  
        # Convert lat and lon differences to radians
        reference_lat_rad = np.radians(reference_lat)
        lon_diff_rad = np.radians(lon_diff)
        # Distance using Haversine formula adjusted by reference latitude
        return R * lon_diff_rad * np.cos(reference_lat_rad)

    def meters_to_lat(self, meters):
        # Earth's radius in meters
        R = 6371000  
        # Inverse calculation for latitude based on Haversine formula
        return np.degrees(meters / R)

    def meters_to_lon(self, meters, reference_lat):
        # Earth's radius in meters
        R = 6371000  
        # Convert reference latitude to radians
        reference_lat_rad = np.radians(reference_lat)
        # Inverse calculation for longitude based on Haversine formula
        return np.degrees(meters / (R * np.cos(reference_lat_rad)))
        
    def get_translations(self, bool_infer_factor, i, best_index, src_pts, dst_pts):
                translation_method = self.translation_method # 0 is phase corr, 1 is direct src normalization svd (rigid with rotation fix), 2 is affine, 3 is rigid svd(no independence, single rot transl estimation), 4 is homography

                
                translation_x, translation_y = 0, 0
                
                if translation_method == 0:
                    pass 

                    # image_to_infer_normed = None
                    # if bool_infer_factor:
                    #     image_to_infer_normed = self.pull_image(i, self.directory, bool_rotate=True, use_estimate=False) # comparison with reference images
                    # else:
                    #     image_to_infer_normed = self.pull_image(i, self.directory, bool_rotate=True, use_estimate=True) # comparison between ref and inference
                    # reference_image_normed = self.pull_image(best_index, self.directory, bool_rotate=True) # ref 

                    # # image_to_infer_normed = cv2.GaussianBlur(image_to_infer_normed, (3, 3), 0)
                    # # reference_image_normed = cv2.GaussianBlur(reference_image_normed, (3, 3), 0) # global matching is highly noise variant
                    # shift, _ = cv2.phaseCorrelate(np.float32(image_to_infer_normed), np.float32(reference_image_normed))
                    
                    # tx1, ty1 = shift
                    # translation_x, translation_y = tx1, ty1

                elif translation_method == 1:
                    # direct src normalization and computation
                    tx2, ty2 = get_src_shifts(src_pts, dst_pts, ret_angle=False)
                    translation_x, translation_y = tx2, ty2
                    
                                                        
                elif translation_method == 2:
                    src_pts = src_pts.reshape(-1,2)
                    dst_pts = dst_pts.reshape(-1,2)
                    # affine: translation, rotation
                    # This is partial affine 2D
                    tx3, ty3, deg3 = ransac_affine_transformation(src_pts, dst_pts)
                    translation_x, translation_y = tx3, ty3
                elif translation_method == 3:
                    # rigid: translation, rotation (manual method)
                    src_pts = src_pts.reshape(-1,2)
                    dst_pts = dst_pts.reshape(-1,2)
                    tx4, ty4, deg4 = rigid_transformation(src_pts, dst_pts)
                    translation_x, translation_y = tx4, ty4
                elif translation_method == 4:
                    # homography: translation, rotation, scale, shear, perspective
                    src_pts = src_pts.reshape(-1,2)
                    dst_pts = dst_pts.reshape(-1,2)
                    tx5, ty5, deg5 = homography_transformation(src_pts, dst_pts)
                    translation_x, translation_y = tx5, ty5
            
                    # translation_x, translation_y = tx2, ty2 # xxx back to 2,3
                
                
                # if not bool_infer_factor:
                #     self.store_and_append_stability(actual_pixel_change_x_m, actual_pixel_change_y_m, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, tx5, ty5)    
            
                return translation_x, translation_y

    def ensure_parallel_lines(self, src_pts, dst_pts, int_angle, tolerance=1.8, max_rotation=180):
        tolerance_adder = -0.5 + 0.5 * len(src_pts)/1000 
        #1.8 works well for 1000 or so. lets expand for larger. 
        # clip to -1, 1
        tolerance_adder = np.clip(tolerance_adder, -1, 1)
        tolerance += tolerance_adder
        
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm != 0 else v
        scaled_tolerance = 2*tolerance * ((1 - 1*(int_angle / max_rotation)) ** 2)
        tolerance = scaled_tolerance
        # Calculate displacement vectors
        displacement_vectors = dst_pts - src_pts

        # Normalize displacement vectors
        normalized_vectors = np.array([normalize(v) for v in displacement_vectors])
        normalized_vectors = np.squeeze(normalized_vectors)  # Ensure shape is (N, 2)

        # Compute the median and mean directions
        median_direction = np.median(normalized_vectors, axis=0)
        combined_direction = normalize(median_direction )

        # print(f"Median direction: {median_direction}")
        # print(f"Mean direction: {mean_direction}")
        # print(f"Weighted Average Direction (3*median + 1*mean): {combined_direction}")
        # print(f"Weighted Average Angle: {np.degrees(np.arctan2(combined_direction[1], combined_direction[0]))}")

        # Calculate angles between each vector and the combined direction
        dot_products = np.dot(normalized_vectors, combined_direction)
        angles = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))  # Get the angle in degrees

        # print(f"Angles above npabs 0, 1, 2, 5, 10, 15: {len(np.where(angles > 0)[0])}, {len(np.where(angles > 1)[0])}, {len(np.where(angles > 2)[0])}, {len(np.where(angles > 5)[0])}, {len(np.where(angles > 10)[0])}, {len(np.where(angles > 15)[0])}")

        # Filter points that deviate by more than the allowed angular tolerance
        filtered_indices = np.where(angles <= tolerance)[0]

        return src_pts[filtered_indices], dst_pts[filtered_indices]

    
    def crop_image(self, img):
        """Crop the top and bottom 10% of the image."""

        # height, width = img.shape[:2]
        # crop_size = int(height * 0.05)  # 10% of the height
        # # cropped_image = cv2.GaussianBlur(cropped_image, (kernel_to_test, kernel_to_test), 0)  # Denoise
        # cropped_image = img[crop_size:height-crop_size, :]
        # # overlay_images(cropped_image, cropped_image)
        # return img[crop_size:height-crop_size, :]  # Crop top and bottom

        height, width = img.shape[:2]
        
        # failed after iteration 3

        target_height = (int)(972)
        target_width = (int)(1920)
        # print(f"Target height: {target_height}, target width: {target_width}")
        
        # Default to cropping 5% from top/bottom and 2% from left/right
        if target_height is None:
            crop_height = int(height * 0.05)  # 5% off top and bottom
        else:
            crop_height = (height - target_height) // 2
        
        if target_width is None:
            crop_width = int(width * 0.02)  # 2% off left and right
        else:
            crop_width = (width - target_width) // 2
        
        # Ensure we don't crop more than the image dimensions
        cropped_image = img[
            max(crop_height, 0):min(height - crop_height, height), 
            max(crop_width, 0):min(width - crop_width, width)
        ]
        
        return cropped_image

    def visualize_pts_and_actual(self, visual_shifts, tx, ty):
        

        normalized_inferred_factors = np.sqrt(self.inferred_factor_x**2 + self.inferred_factor_y**2)

        # Check if visual_shifts is 1D and reshape it
        print(f"Visual shifts shape: {visual_shifts.shape}")
        if visual_shifts.ndim != 2:
            visual_shifts = visual_shifts.reshape(-1, 2)

        # Calculate the radial distance (norm) for the visual shifts
        vis_x = visual_shifts[:, 0]
        vis_y = visual_shifts[:, 1]
        vis_x = vis_x * self.inferred_factor_x
        vis_y = vis_y * self.inferred_factor_y
        visual_shifts = np.column_stack((vis_x, vis_y))
        visual_norms = np.linalg.norm(visual_shifts, axis=1)
        
        # Calculate the radial distance (norm) for the actual translation
        actual_norm = np.sqrt(tx**2 + ty**2)
        actual_norm = actual_norm 

        # Plot histogram for the radial shifts (norms)
        plt.figure(figsize=(8, 6))
        plt.hist(visual_norms, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(actual_norm, color='red', linestyle='--', label=f'Actual norm = {actual_norm:.2f}')
        plt.title('Histogram of Radial Visual Shifts (Norms)')
        plt.xlabel('Radial shift (norm)')
        plt.ylabel('Frequency')
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_matches(self, img1, img2, kp1, kp2, good_matches):
    # Create a new image to hold both images side by side
        if len(img1.shape) == 2:  # Check if img1 is grayscale
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:  # Check if img2 is grayscale
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        # Place the first image on the left
        combined_image[:h1, :w1] = img1
        
        # Place the second image on the right
        combined_image[:h2, w1:w1+w2] = img2
        # sort matches only top 50
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        good_matches = good_matches[:50]
        bad_matches = sorted(good_matches, key=lambda x: x.distance, reverse=True)
        bad_matches = bad_matches[:50]
        # Loop through the good matches to draw lines
        for match in good_matches:
            # Get the keypoint coordinates from both images
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            # Scale the second point to align with the combined image
            pt2 = (pt2[0] + w1, pt2[1])  # Shift x-coordinate to the right image

            # Draw the line connecting the two points
            cv2.line(combined_image, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), 
                    (0, 255, 0), 1)  # Green line

            # Draw circles at each keypoint
            cv2.circle(combined_image, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 0), -1)  # Blue circle for img1
            cv2.circle(combined_image, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 0), -1)  # Blue circle for img2

        # Show the combined image with matches
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.title('Top 50 Matched Keypoints')
        plt.show()

    def filter_by_gradient(self, src_pts, dst_pts, tolerance=3, use_median=False):
        src_pts = src_pts.reshape(-1, 2)
        dst_pts = dst_pts.reshape(-1, 2)
        if use_median:
            expected_gradient = np.median((dst_pts[:, 1] - src_pts[:, 1]) / (dst_pts[:, 0] - src_pts[:, 0]))
        else:
            expected_gradient = np.mean((dst_pts[:, 1] - src_pts[:, 1]) / (dst_pts[:, 0] - src_pts[:, 0]))
        
        actual_gradients = (dst_pts[:, 1] - src_pts[:, 1]) / (dst_pts[:, 0] - src_pts[:, 0])
        angle_differences = np.abs(np.degrees(np.arctan(actual_gradients)) - np.degrees(np.arctan(expected_gradient)))
        angle_differences = np.clip(angle_differences, 0, 180)
        
        filtered_indices = np.where(angle_differences <= tolerance)[0]
        
        return src_pts[filtered_indices], dst_pts[filtered_indices]

    def remove_out_of_stdev(self, src_pts, dst_pts, std_devs=2):

        src_pts = src_pts.reshape(-1,2)
        dst_pts = dst_pts.reshape(-1,2)
        # Calculate the mean and standard deviation of the shifts
        mean = np.mean(dst_pts - src_pts, axis=0)
        std_dev = np.std(dst_pts - src_pts, axis=0)
        # Filter out points that deviate by more than the allowed standard deviations
        mask = np.all(np.abs(dst_pts - src_pts - mean) <= std_devs * std_dev, axis=1)
        filtered_src_pts = src_pts[mask]
        filtered_dst_pts = dst_pts[mask]
        # next, reshape back 
        filtered_src_pts = filtered_src_pts.reshape(-1, 1, 2)
        filtered_dst_pts = filtered_dst_pts.reshape(-1, 1, 2)
        return filtered_src_pts, filtered_dst_pts

    def gps_diff_haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000  # Earth's radius in meters

        # Convert latitudes and longitudes from degrees to radians
        lat1_rad, lon1_rad = np.radians([lat1, lon1])
        lat2_rad, lon2_rad = np.radians([lat2, lon2])

        # Differences in latitudes and longitudes
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula for latitudinal distance (along meridian)
        lat_a = np.sin(dlat / 2) ** 2
        lat_c = 2 * np.arctan2(np.sqrt(lat_a), np.sqrt(1 - lat_a))
        lat_diff_meters = R * lat_c

        # Haversine formula for longitudinal distance (along parallel at given latitude)
        a = np.sin(dlon / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        lon_c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        lon_diff_meters = R * lon_c

        return lat_diff_meters, lon_diff_meters  # Return as a tuple

    def meters_to_gps_haversine(self, lat, lon, translation_x_m, translation_y_m):
        R = 6371000  # Earth's radius in meters

        # Convert latitude and longitude from degrees to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Calculate the displacement in angular distance (radians)
        angular_distance_lat = translation_y_m / R
        angular_distance_lon = translation_x_m / R

        # Compute the new latitude (formula for latitude translation along a great circle)
        new_lat_rad = np.arcsin(np.sin(lat_rad) * np.cos(angular_distance_lat) +
                                np.cos(lat_rad) * np.sin(angular_distance_lat))

        # Compute the new longitude (formula for longitude translation along a great circle)
        delta_lon = np.arctan2(
            np.sin(angular_distance_lon) * np.cos(lat_rad),
            np.cos(angular_distance_lat) - np.sin(lat_rad) * np.sin(new_lat_rad)
        )
        new_lon_rad = lon_rad + delta_lon

        # Convert the new latitude and longitude from radians back to degrees
        new_lat = np.degrees(new_lat_rad)
        new_lon = np.degrees(new_lon_rad)

        return new_lat, new_lon








    def analyze_matches(self, bool_infer_factor, num_images_analyze):
        deviation_norms_x = []
        deviation_norms_y = []

        range_im = num_images_analyze
        
        for i in reversed(range(1, range_im)):  # iterate through all images in reverse order
            # print(f"Analyzing image {i}...")
            if i==1:
                continue
            START_TIME = time.time()
            best_index = -1   
            internal_angle = None
            # image_space = self.find_image_space(i)
            image_space = self.find_closest_images(self.stored_gps[i], i)
            
            if len(image_space) == 0:
                print(f"No images found in the vicinity of image {i}. Skipping.")
                continue
            timeAS = time.time()
            match_time_1 = time.time()
            if self.global_matching_technique < 3:
                best_index, internal_angle = self.find_best_match(i, image_space)           
              
            else: 
                best_index, internal_angle = self.find_best_match_multimethod(i, image_space)
            match_time_2 = time.time() - match_time_1
            
            if best_index != -1 and internal_angle is not None:

                time_rand = time.time()
                
                # please note: the images in this dataset have assumed headings based on testing, as such they are not exact, and images that were meant to have perfect reference headings have estimated ones, thus adding a partial error to the system. 

                # actual GPS data to compare against
                actual_gps_diff_meters = (np.array(self.stored_gps[i]) - np.array(self.stored_gps[best_index])) #* 111139

                actual_gps_diff_meters = (
                    self.lon_to_meters(actual_gps_diff_meters[0], self.stored_gps[i][1]),  # Long difference in meters (Y axis)
                    self.lat_to_meters(actual_gps_diff_meters[1])  # Lat difference in meters (X axis)
                )

                actual_pixel_change_x_m, actual_pixel_change_y_m = actual_gps_diff_meters[0], actual_gps_diff_meters[1]

                # 0

                self.actual_GPS_deviation.append((np.abs(actual_pixel_change_x_m), np.abs(actual_pixel_change_y_m)))

                if bool_infer_factor:
                    inference_image = self.pull_image(i, self.directory, bool_rotate=False)
                    inf_kp, inf_des = self.stored_local_keypoints[i], self.stored_local_descriptors[i] if self.neural_net_on == False else (self.stored_feats[i], None)
                elif not bool_infer_factor:
                    inference_image = self.pull_image(i, self.directory, bool_rotate=False)
                    inf_kp, inf_des = self.local_detector.get_keydes(inference_image) if self.local_detector_name != "SuperPoint" else (self.local_detector.get_features(inference_image), None)
                
                rand_time_2 = time.time() - time_rand

                if bool_infer_factor:
                    # print(f"len feats: {len(self.stored_feats[i]['keypoints'])}") if self.neural_net_on == True 
                    src_pts, dst_pts, _ = self.get_src_dst_pts(inf_kp, self.stored_local_keypoints[best_index], inf_des, self.stored_local_descriptors[best_index], 0.8, global_matcher_true=False) if self.neural_net_on == False else get_neural_src_pts(inf_kp, self.stored_feats[best_index])
                elif not bool_infer_factor:
                    src_pts, dst_pts, _ = self.get_src_dst_pts(inf_kp, self.stored_local_keypoints[best_index], inf_des, self.stored_local_descriptors[best_index], 0.8, global_matcher_true=False) if self.neural_net_on == False else get_neural_src_pts(inf_kp, self.stored_feats[best_index])




                extra_internal_angle = get_src_shifts(src_pts, dst_pts, ret_angle=True) 
                # extra_internal_angle = 0
                inference_image_rotated = rotate_image(inference_image, extra_internal_angle)
                
                
                rotated_inf_kp, rotated_inf_des = self.local_detector.get_keydes(inference_image_rotated) if self.local_detector_name != "SuperPoint" else (self.local_detector.get_features(inference_image_rotated), None)
                

                if bool_infer_factor:
                    # print(f"len feats: {len(self.stored_feats[i]['keypoints'])}") if self.neural_net_on == True 
                    src_pts, dst_pts, _ = self.get_src_dst_pts(rotated_inf_kp, self.stored_local_keypoints[best_index], rotated_inf_des, self.stored_local_descriptors[best_index], 0.8, global_matcher_true=False) if self.neural_net_on == False else get_neural_src_pts(rotated_inf_kp, self.stored_feats[best_index])
                elif not bool_infer_factor:
                    src_pts, dst_pts, gd_matches = self.get_src_dst_pts(rotated_inf_kp, self.stored_local_keypoints[best_index], rotated_inf_des, self.stored_local_descriptors[best_index], 0.8, global_matcher_true=False) if self.neural_net_on == False else get_neural_src_pts(rotated_inf_kp, self.stored_feats[best_index])
                    # self.plot_matches(inference_image_rotated, self.pull_image(best_index, self.directory, bool_rotate=False), rotated_inf_kp, self.stored_local_keypoints[best_index], gd_matches)
                

                # if not bool_infer_factor:
                #     visual_shifts = dst_pts - src_pts
                #     self.visualize_pts_and_actual(visual_shifts, actual_pixel_change_x_m, actual_pixel_change_y_m)
                prior_src, prior_dst = src_pts, dst_pts
                src_pts, dst_pts = self.remove_out_of_stdev(src_pts, dst_pts, 2)
                src_pts, dst_pts = self.filter_by_gradient(src_pts, dst_pts, 0.5, use_median=True) 
                # src_pts, dst_pts = self.ensure_parallel_lines(src_pts, dst_pts, np.abs(internal_angle)) # INSTABILITY 
                if (len(src_pts) < 200):
                    src_pts, dst_pts = prior_src, prior_dst
                # print(f"len src pts: {len(src_pts)}")





                # if not bool_infer_factor:
                #     visual_shifts = dst_pts - src_pts
                #     self.visualize_pts_and_actual(visual_shifts, actual_pixel_change_x_m, actual_pixel_change_y_m)

                rand_time_3 = time.time() - time_rand
                # debug (end of analyze)
                len_matches = len(src_pts)
                self.len_matches_arr.append(len_matches)

                        

                shift_time_1 = time.time()
                translation_x, translation_y = self.get_translations(bool_infer_factor, i, best_index, src_pts, dst_pts)
                new_diff = (dst_pts - src_pts).reshape(-1, 2)
                # # calculate median translation
                
                
                shift_time_2 = time.time() - shift_time_1

                # DEBUG
                Unnorm_x, Unnorm_y = translation_x, translation_y
                rand_new = time.time() # NOTHING 1
                # Global Normalization
                translation_x, translation_y = normalize_translation_to_global_coord_system(translation_x, translation_y, -self.stored_headings[best_index])

                # INFERENCE
                if bool_infer_factor and actual_pixel_change_x_m != 0 and actual_pixel_change_y_m != 0:
                    self.append_answers_estimates(translation_x, translation_y, actual_pixel_change_x_m, actual_pixel_change_y_m)
    
                # pixel -> meters (estimated)
                translation_x_m = translation_x * self.inferred_factor_x 
                translation_y_m = translation_y * self.inferred_factor_y

                # NEW GPS (estimated). Conversion of metres to GPS and summing to reference GPS. 
                # new_lon = self.stored_gps[best_index][0] + self.meters_to_lon(translation_x_m, self.stored_gps[best_index][1])
                # new_lat = self.stored_gps[best_index][1] + self.meters_to_lat(translation_y_m)
                new_lat, new_lon = self.meters_to_gps_haversine(
                    self.stored_gps[best_index][1],  # Original latitude
                    self.stored_gps[best_index][0],  # Original longitude
                    translation_x_m,                 # Translation in meters along X (longitude)
                    translation_y_m                  # Translation in meters along Y (latitude)
                )
                if not bool_infer_factor:
                    self.estimated_gps.append((new_lon, new_lat))

                rand_time_half = time.time() # NOTHING 2
                # DEBUG
                deviation_x_meters = translation_x_m - actual_pixel_change_x_m
                deviation_y_meters = translation_y_m - actual_pixel_change_y_m
                deviation_norms_x.append(np.abs(deviation_x_meters))
                deviation_norms_y.append(np.abs(deviation_y_meters))
                rand_new_2 =  time.time() - rand_time_half 
                # print(f"DEV-X,Y (m): {deviation_x_meters}, {deviation_y_meters}  for im {i+1} wrt {best_index+1}, angle: {((self.stored_headings[i])-(self.estimated_headings[i])):.4f} deg, actual deviation (m): {actual_pixel_change_x_m/4.8922887}, {actual_pixel_change_y_m/4.25715961}")

                

                image_size = self.pull_image(i, self.directory, bool_rotate=False).shape
                act_tx_ratio = 1 - np.abs(Unnorm_x) / image_size[1]
                act_ty_ratio = 1 - np.abs(Unnorm_y) / image_size[0]
                act_tx_pixels = act_tx_ratio*image_size[1]
                act_ty_pixels = act_ty_ratio*image_size[0]
                # print(f"unnorm xy: {Unnorm_x}, {Unnorm_y}")
                act_tx_pixels = image_size[1] - np.abs(Unnorm_x)
                act_ty_pixels = image_size[0] - np.abs(Unnorm_y)
                act_tx_percent = act_tx_ratio * 100
                act_ty_percent = act_ty_ratio * 100
                # print(f"X-overlap: {act_tx_percent:.2f}%, X-dev: {deviation_x_meters:.2f}m, Y-overlap: {act_ty_percent:.2f}%, Y-dev: {deviation_y_meters:.2f}m ")
                if not bool_infer_factor:
                    self.x_overlap.append(act_tx_pixels)
                    self.y_overlap.append(act_ty_pixels)
                x_disp_ratio = np.abs(Unnorm_x) / image_size[1]
                y_disp_ratio = np.abs(Unnorm_y) / image_size[0]
                total_overlap_ratio = x_disp_ratio * y_disp_ratio # not right - diff dims 

                # we want to plot the entire path. so lets output the estimated and actual GPS coordinates.
                # if not bool_infer_factor:
                #     print(f"Estimated GPS: {new_lon:.9f}, {new_lat:.9f}, Actual GPS: {self.stored_gps[i][0]:.9f}, {self.stored_gps[i][1]:.9f}")

                # lets do the same print but with pixel estimates 
                
                if not bool_infer_factor:
                    # Get the actual GPS coordinates in meters using Haversine
                    actual_lat_diff_meters, actual_lon_diff_meters = self.gps_diff_haversine(
                        self.stored_gps[i][1], self.stored_gps[i][0],  # lat1, lon1
                        new_lat, new_lon  # lat2, lon2
                    )

                    # Convert the GPS differences to pixel differences
                    actual_pixels = actual_lon_diff_meters / self.inferred_factor_x, actual_lat_diff_meters / self.inferred_factor_y

                    # Calculate new meters from new lat/lon using Haversine again
                    new_lat_diff_meters, new_lon_diff_meters = self.gps_diff_haversine(
                        new_lat, new_lon,
                        self.stored_gps[i][1], self.stored_gps[i][0]
                    )

                    # Convert the new meters into estimated pixels
                    estimated_pixels = new_lon_diff_meters / self.inferred_factor_x, new_lat_diff_meters / self.inferred_factor_y
                    difference_in_pixels = np.abs(estimated_pixels[0] - actual_pixels[0]), np.abs(estimated_pixels[1] - actual_pixels[1])
                    difference_est_act = deviation_x_meters, deviation_y_meters
                    # find st.dev, mean of dst-src
                    new_diff_x = (dst_pts - src_pts).reshape(-1, 2)[:, 0]
                    new_diff_y = (dst_pts - src_pts).reshape(-1, 2)[:, 1]

                    stdev_x, stdev_y = np.std(new_diff_x), np.std(new_diff_y)
                    mean_x, mean_y = np.mean(new_diff_x), np.mean(new_diff_y)
                    median_x, median_y = np.median(new_diff_x), np.median(new_diff_y)

                    # print(f"Estimated pixels: {estimated_pixels[0]:.9f}, {estimated_pixels[1]:.9f}, Actual pixels: {actual_pixels[0]:.9f}, {actual_pixels[1]:.9f}")
                    # compare internal angle to pixel error:
                    # get difference between estimated internal angle and actual internal angle
                    # diff_angle = np.abs(extra_internal_angle - (self.stored_headings[i] - self.stored_headings[best_index]))
                    
                    # if not bool_infer_factor:
                    #     pass
                        

                        # print(f"Interm angle: {-self.stored_headings[best_index]:.4f} deg, DEV-X,Y (pixels): {difference_est_act}")
                        # print(f"AngleDiff: {diff_angle:.4f} deg, DEV-X,Y (pixels): {difference_in_pixels}")
                        # print(f"mean vs med vs st.dev vs error (m): {mean_x:.2f}, {mean_y:.2f}, {median_x:.2f}, {median_y:.2f}, {stdev_x:.2f}, {stdev_y:.2f}, {difference_in_pixels}")

                    
                END_TIME = time.time() - START_TIME
                if bool_infer_factor:
                    self.parameter_inference_time_arr.append(END_TIME) # note this is in reverse
                elif not bool_infer_factor:
                    self.location_inference_time_arr.append(END_TIME)


                
                time_BB = time.time() - timeAS
                
                # print("Match time: ", match_time_2, "Shift time: ", shift_time_2, "Total time: ", time_BB, "Rand time: ", rand_time_2, "Rand time 2: ", rand_time_3, "Rand new 2: ", rand_new_2, "Rand new: ", rand_time_half)

            else:
                print(f"Could not find a match for image {i}. Skipping...")
        mean_x_dev, mean_y_dev = np.mean(deviation_norms_x), np.mean(deviation_norms_y)
        # if not bool_infer_factor:
        #     self.print_stability_analysis()
        


        total_normalized_gps_error = np.linalg.norm([mean_x_dev, mean_y_dev])
        MAE_GPS = mean_x_dev/2 + mean_y_dev/2
        if not bool_infer_factor:
            self.MAE_GPS.append(MAE_GPS)
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


def normalize_images_to_global_coord_system(ref_img, inference_image, reference_heading, internal_angle):
    rotation_angle = -internal_angle#-estimate_affine_rotation(inference_image, ref_img)
    rot_inf_img = rotate_image(inference_image, rotation_angle + reference_heading)
    rot_ref_img = rotate_image(ref_img, reference_heading)
    return rot_inf_img, rot_ref_img




def normalize_translation_to_global_coord_system(tx, ty, reference_heading, internal_angle=0):
    # The same rotation angle logic as the original function
    theta_inf = reference_heading #- internal_angle
    
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
    global_detector_choice = 1  # # Set 1 for ORB, 2 for AKAZE
    global_matcher_choice = 1  # Set 0 for BFMatcher, 1 for FlannMatcher, 2 for GraphMatcher
    global_matcher_technique = 3  # Set 0 for Pass Through grid counter, 3 for correlation, 4 for histogram, 5 for SSIM, 6 for Hash (DONT USE HASHING). 
    local_detector_choice = 2  # Set 1 for ORB, 2 for AKAZE 3 for NEURAL (lightglue matcher)
    local_matcher_choice = 1  # Set 0 for BFMatcher, 1 for FlannMatcher, 2 for GraphMatcher
    


    # rotation
    rotational_detector_choice = 1  # Set 1 for ORB, 2 for AKAZE'
    rotation_method_to_use = 2 # 0 for affine, 1 for homography, 2 for partial affine

    # translation
    translation_method_to_use = 1 # 0 for phase correlation, 1 for SVD rigid (fix rot), 2 for affine, 3 for partial 2D, 4 for homography
   



    # DATASET optimization
    main_dataset_name = "DATSETROCK" 
    message = "EFFICIENCYMODE\n"

    # AKAZE thresh, ORB kp. 

    num_images = 15
    inference_images = 7

    global_detector_arr = [1,2]
    global_matcher_arr = [0,1,2]
    #ROT, CPT, ROCK, SAND, AMAZ - ALL: DATSETXXXX
    dat_set_arr = ["DATSETROT", "DATSETCPT", "DATSETROCK", "DATSETSAND", "DATSETAMAZ"]
    # dat_set_arr = ["DATSETROCK"]
   
   
    global_matcher_technique_arr = [2,4]
    # local_detector_arr = [1,2, 3]
    local_detector_arr = [3]
    local_matcher_arr = [0,1,2]
    iteration_count = 0
    

    rotation_method_to_use_arr = [0,1,2]
    
    if 1==1:
    
        # for global_matcher_choice in global_matcher_arr:
            # for translation_method_to_use in [2]:
                # for local_detector_choice in local_detector_arr:
            # for rotation_method_to_use in rotation_method_to_use_arr:
                #     for global_detector_choice in global_detector_arr:
                # for scale_factor in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
                # for local_matcher_choice in local_matcher_arr:
                    for main_dataset_name in dat_set_arr:
                        
                    # for global_matcher_technique in global_matcher_technique_arr:
    #                
    #                     # We only want to run one instance of neural mode, others are redundant.
    #                     if local_detector_choice == 3 and local_matcher_choice !=0
    #                         continue
                        # if the navigator object exists
                        directory = './GoogleEarth/DATASETS/{}'.format(main_dataset_name)
                        

                        if 'navigator' in locals():
                            # navigator.reset_all_data()
                            del navigator
                            gc.collect()
                            print(f'End of Iteration: {iteration_count}') 
                        print(f"Dataset: {main_dataset_name}")



                        glob_thresh=0.00017 if global_detector_choice == 2 else 6000 if global_detector_choice == 1 else 0
                        loc_det_thresh = 0.005 if local_detector_choice == 2 else 6000 if local_detector_choice == 1 else 0
                        rot_det_thresh = 0.0002 if rotational_detector_choice == 2 else 8000 if rotational_detector_choice == 1 else 0                         

                                

                        main_start_time = time.time()
                        
                        navigator = UAVNavigator(global_detector_choice, local_detector_choice , rotational_detector_choice, global_matcher_choice, local_matcher_choice, global_matcher_technique, main_dataset_name, rotation_method_to_use, glob_thresh, loc_det_thresh, rot_det_thresh, translation_method_to_use) # INITIALIZATION
                        
                        

                        iteration_count += 1
                        
                        # Step 1: Add images and infer factors
                        
                        for i in range(1, inference_images + 1): # stops at inference_images = 6]
                            navigator.add_image(i, directory)
                        navigator.analyze_matches(True, inference_images)
                        navigator.compute_linear_regression_factors()
                        
                        # step 2: Add rest of images when GPS available
                        for i in range(inference_images+1, num_images + 1):
                            navigator.add_image(i, directory)
                        
                    

                        # step 3: Infer pose while NO GPS 
                        navigator.analyze_matches(False, num_images) #  BOOL INFER FACTOR = FALSE. 









                        #DEBUG ONLY
                        # mean_mut_info_x = np.mean(navigator.x_overlap)
                        # mean_retained_info_x = (1920 - 100*scale_factor)/1920
                        # net_information_mutual_x = mean_mut_info_x * mean_retained_info_x
                        # mean_mut_info_y = np.mean(navigator.y_overlap)
                        # mean_retained_info_y = (972 - 100*scale_factor)/972
                        # net_information_mutual_y = mean_mut_info_y * mean_retained_info_y
                        # net_information_mutual_x= 1920 - 1000 - 100*scale_factor
                        # net_information_mutual_y = 972 - 300 - 80*scale_factor

                        # print(f"x, y-overlap-mean: {int(net_information_mutual_x)}, {int(net_information_mutual_y)}")
                        # seperate add_time_arr, parameter_inference_time_arr, and location_inference_time_arr. Get the mean and variance of each
                        string_time_analysis_mean = f"Mean_Add_Time: {np.mean(navigator.add_time_arr)}, Mean_Parameter_Inference_Time: {np.mean(navigator.parameter_inference_time_arr)}, Mean_Location_Inference_Time: {np.mean(navigator.location_inference_time_arr)}"
                        string_time_analysis_var = f"Var_Add_Time: {np.var(navigator.add_time_arr)}, Var_Parameter_Inference_Time: {np.var(navigator.parameter_inference_time_arr)}, Var_Location_Inference_Time: {np.var(navigator.location_inference_time_arr)}"
                        string_total_time = f"Total Time: {np.sum(navigator.add_time_arr) + np.sum(navigator.parameter_inference_time_arr) + np.sum(navigator.location_inference_time_arr)}"
                        print(string_time_analysis_mean)
                        print(string_time_analysis_var)
                        print(string_total_time)


                        #conv np arr


                        np_act_change = np.array(navigator.actual_GPS_deviation)
                        np_est_dev = np.array(navigator.norm_GPS_error)
                        np_MAE_GPS = np.array(navigator.MAE_GPS)
                        # find radial mean movement of UAV
                        gps_act_x, gps_act_y = np.mean(np_act_change[:,0]), np.mean(np_act_change[:,1])
                        mean_radial_movement = np.sqrt(gps_act_x**2 + gps_act_y**2)
                        # find swing percentage
                        swing_percent = np.array(100*np_est_dev/mean_radial_movement)
                        print(f"Percentage Deviation: {swing_percent} %")
                        string_percent_GPS_dev = f"Percentage GPS Deviation: {swing_percent} %"
                        string_params = print_choices(global_detector_choice,global_matcher_choice, global_matcher_technique, local_detector_choice, local_matcher_choice)
                        string_GPS_error = f"RMSE GPS error: {np_est_dev}" 
                        print(f"Mean Absolute Error GPS: {np_MAE_GPS}")
                        String_RMSE = f"MAE GPS error: {np_est_dev}"
                        print(np_est_dev)
                        # string_heading_error = f"Mean Heading Error: {np.mean(navigator.estimated_heading_deviations)}"
                        # print(string_GPS_error, '\n', string_heading_error)
                        




                        num_global_keypoints_per_image = [len(kps) for kps in navigator.stored_global_keypoints] # XXX sort out
                                              
                        string_mean_glob_len_kp = f"Mean Length of Global Keypoints: {np.mean(num_global_keypoints_per_image)}"
                        print(string_mean_glob_len_kp) # XXX append this later to file
                        
                        # num_local_keypoints_per_image = [len(kps) for kps in navigator.stored_local_keypoints] if local_detector_choice != 3 else [len(kps) for kps in navigator.stored_feats[0]['keypoints']]
                        # string_mean_len_local_kp = f"Mean Length of Local Keypoints: {np.mean(num_local_keypoints_per_image)}"
                        # print(string_mean_len_local_kp) # XXX append this later to file
                        # max_min_loc_range_kp = np.max(num_local_keypoints_per_image) - np.min(num_local_keypoints_per_image)
                        # string_max_min_range_matches = f"Range of Local kp: {max_min_loc_range_kp}"
                        # print(string_max_min_range_matches) # XXX append this later to file
                       
                        
                        string_mean_time_kp = f"Mean Global Time to Extract Keypoints: {navigator.keypoint_time/navigator.keypoint_iterations:.4f} s"
                        # print(string_mean_time_kp) # XXX append this later to file
                        mean_good_matches = np.mean(navigator.glob_mat_len_arr)
                        string_mean_good_matches = f"Mean Number of Global good Matches: {mean_good_matches}"
                        mean_loc_good_matches = np.mean(navigator.loc_mat_len_arr)
                        string_mean_loc_good_matches = f"Mean Number of Loc good Matches: {mean_loc_good_matches}"
                        print(string_mean_loc_good_matches) # XXX append this later to file

                        print(string_mean_good_matches) # XXX append this later to file

                        
                        append_to_file("results.txt", string_params, string_GPS_error, string_percent_GPS_dev, message, string_time_analysis_mean, string_time_analysis_var, string_total_time, main_dataset_name, "\n\n")


                        elapsed_time = time.time() - main_start_time 
                        print(f"Time taken to execute The Method: {elapsed_time:.4f} seconds")
                        

                    
    # ANY code outside of all iterations. 
    


if __name__ == "__main__":
    main() 
