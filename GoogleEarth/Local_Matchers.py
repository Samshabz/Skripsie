import numpy as np
import cv2


# best matchers: (this is plussed by 1) 8 is 6, 13 is 11, 11 is 9 


# Base class for matchers
class BaseMatcher:
    def find_matches(self, des1, des2, kp1, kp2, detector_choice):
        raise NotImplementedError("This method should be overridden by subclasses")



# BFMatcher class for binary descriptors
class BFMatcher(BaseMatcher):
    def __init__(self):
        

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def find_matches(self, des1, des2, kp1=0, kp2=0, detector_choice=0, global_matcher_true=0):
        # Remove batch dimension if present

        # Perform matching  
        matches = self.matcher.knnMatch(des1, des2, k=2) 
        return matches

# FLANN Matcher class for binary descriptors
class FlannMatcher(BaseMatcher):
    def __init__(self):
        # LSH parameters for binary descriptors like ORB and AKAZE
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6,  # Higher table number gives better accuracy (range: 10-30)
                            key_size=6,      # Key size; 20 is optimal for accuracy (range: 10-30)
                            multi_probe_level=1)  # Higher values increase accuracy but reduce speed (range: 1-2)
        
        search_params = dict(checks=50)  # Increase this for more exhaustive search (range: 100-1000)
        
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def find_matches(self, des1, des2, kp1, kp2, detector_choice, global_matcher_true):
        # Check if descriptors are valid and have enough keypoints
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []  # No matches if not enough descriptors

        # KNN Matching
        return self.matcher.knnMatch(des1, des2, k=2)

# LSH Matcher for binary descriptors


# Graph Matcher using RANSAC for geometric consistency
class GraphMatcher(BaseMatcher):
    def __init__(self):

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def find_matches(self, des1, des2, kp1, kp2, detector_choice, global_matcher_true):
        # Check if descriptors and keypoints are valid
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return []  # No matches if not enough descriptors or keypoints

        # BFMatcher to find initial matches
        matches = self.matcher.match(des1, des2)
        
        if len(matches) < 4:
            return []  # Not enough matches for homography calculation

        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Apply RANSAC to eliminate outliers and find homography
        homography_threshold = 1
        if detector_choice == 1: # this is orb. requires high threshold since noisy matches
            homography_threshold = 25.0 # 35 is super acc but way too slow
        elif detector_choice == 2:
            homography_threshold = 1.0 if global_matcher_true != 1 else 0.25
            # the else is for when we use the global matcher which grids images so we want this to be super fast, ie a lower threshold which allows for more matches
            # lower for AKAZE as it has robust and accurate matches
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_threshold)

        # Filter good matches based on the mask
        good_matches = [m for i, m in enumerate(matches) if mask[i]]
        return good_matches




# Set matcher function to return the correct matcher object
def set_matcher(matcher_choice):
    """
    Returns the correct matcher based on the user's choice.
    Supported options: "bf_matcher", "flann_matcher", "lsh_matcher", "ann_matcher", "graph_matcher"
    """
    if matcher_choice == "bf_matcher":
        return BFMatcher()
    elif matcher_choice == "flann_matcher":
        return FlannMatcher()
    elif matcher_choice == "graph_matcher":
        return GraphMatcher()

    else:
        raise ValueError(f"Invalid matcher choice: {matcher_choice}")
