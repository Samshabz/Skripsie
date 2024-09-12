import cv2

# Base class for matchers
class BaseMatcher:
    def find_matches(self, des1, des2):
        raise NotImplementedError("This method should be overridden by subclasses")

# BFMatcher class for binary descriptors
class BFMatcher(BaseMatcher):
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def find_matches(self, des1, des2):
        # KNN Matching
        return self.matcher.knnMatch(des1, des2, k=2)

# FLANN Matcher class for binary descriptors
class FlannMatcher(BaseMatcher):
    def __init__(self):
        # LSH parameters for binary descriptors like ORB and AKAZE
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=20,  # Higher table number gives better accuracy (range: 10-30)
                            key_size=20,      # Key size; 20 is optimal for accuracy (range: 10-30)
                            multi_probe_level=2)  # Higher values increase accuracy but reduce speed (range: 1-2)
        
        search_params = dict(checks=500)  # Increase this for more exhaustive search (range: 100-1000)
        
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def find_matches(self, des1, des2):
        # Check if descriptors are valid and have enough keypoints
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []  # No matches if not enough descriptors

        # KNN Matching
        return self.matcher.knnMatch(des1, des2, k=2)

# LSH Matcher for binary descriptors

# ANN Matcher using FLANN for fast approximate matching
class ANNMatcher(BaseMatcher):
    def __init__(self):
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def find_matches(self, des1, des2):
        # KNN Matching
        return self.matcher.knnMatch(des1, des2, k=2)

# Graph Matcher using RANSAC for geometric consistency
class GraphMatcher(BaseMatcher):
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def find_matches(self, des1, des2):
        matches = self.matcher.match(des1, des2)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Apply RANSAC to find homography and eliminate outliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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
    elif matcher_choice == "ann_matcher":
        return ANNMatcher()
    elif matcher_choice == "graph_matcher":
        return GraphMatcher()
    else:
        raise ValueError(f"Invalid matcher choice: {matcher_choice}")
