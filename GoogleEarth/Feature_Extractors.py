
import cv2
# Base class for feature extractors
class BaseFeatureExtractor:
    def get_keydes(self, image):
        """
        Extracts keypoints and descriptors from the image.
        Needs to be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def get_features(self, image):
        """
        Extracts features only from the image (for light matching).
        Needs to be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

# ORB Feature Extractor
class ORBFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=20000)

    def get_keydes(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        return keypoints, descriptors

    def get_features(self, image):
        keypoints, _ = self.detector.detectAndCompute(image, None)
        return keypoints

# AKAZE Feature Extractor
class AKAZEFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.detector = cv2.AKAZE_create(threshold=0.0005)

    def get_keydes(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def get_features(self, image):
        keypoints, _ = self.detector.detectAndCompute(image, None)
        return keypoints



# Set extractor function to return the correct feature extractor object
def set_feature_extractor(detector_choice, device=None):
    """
    Returns the correct feature extractor based on the user's choice.
    Supported options: 1 for ORB, 2 for AKAZE, 3 for SuperPoint.
    """
    if detector_choice == 1:
        return ORBFeatureExtractor()
    elif detector_choice == 2:
        return AKAZEFeatureExtractor()
    else:
        raise ValueError(f"Invalid detector choice: {detector_choice}")
