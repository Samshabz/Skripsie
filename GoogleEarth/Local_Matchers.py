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
        return self.matcher.knnMatch(des1, des2, k=2) 



# FLANN Matcher class for binary descriptors
class FlannMatcher(BaseMatcher):
    def __init__(self):
        # LSH parameters for binary descriptors like ORB and AKAZE
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=2,  # Lower table number (6-10) for faster but less accurate search
                            key_size=10,     # Lower key size (10-15) for faster search
                            multi_probe_level=1)  # Increase multi-probe to 2 for slight accuracy boost
        
        search_params = dict(checks=2)  # Reduce checks for faster but less exhaustive search

        
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def find_matches(self, des1, des2, kp1=0, kp2=0, detector_choice=0, global_matcher_true=0):
        return self.matcher.knnMatch(des1, des2, k=2)



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

    else:
        raise ValueError(f"Invalid matcher choice: {matcher_choice}")
