
import cv2
import torch
from lightglue import SuperPoint

# Base class for feature extractors - algorithmic
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
    def __init__(self, nfeatures=3000):
        self.detector = cv2.ORB_create(nfeatures=nfeatures)

    def get_keydes(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        return keypoints, descriptors

    def get_features(self, image):
        keypoints, _ = self.detector.detectAndCompute(image, None)
        return keypoints

# AKAZE Feature Extractor
class AKAZEFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, threshold=0.0001):
        self.detector = cv2.AKAZE_create() # Lower implies more keypoints.0052
        
        self.detector.setThreshold(threshold)
    # higher is more aggressive filtering. 
    def get_keydes(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def get_features(self, image):
        keypoints, _ = self.detector.detectAndCompute(image, None)
        return keypoints




def set_feature_extractor(detector_choice, threshold=0.0001, device=None):
    """
    Returns the correct feature extractor based on the user's choice.
    Supported options: 1 for ORB, 2 for AKAZE, 3 for SuperPoint.
    """
    if detector_choice == 1:
        return ORBFeatureExtractor(threshold)
    elif detector_choice == 2:
        if threshold !=0.0001:
            return AKAZEFeatureExtractor(threshold)
        else:
            return AKAZEFeatureExtractor()
    else:
        raise ValueError(f"Invalid detector choice: {detector_choice}")


# SuperPoint Feature Extractor
class BaseFeatureExtractorNEURAL:
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

class SuperPointFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, device):
        self.device = device
        self.extractor = SuperPoint(
            max_num_keypoints=1024, #2048,
            # detection_threshold=0.0000015,# lower implies more keypoints - prior: 0000015
            # nms_radius=5
        ).eval().to(device)

    def normalize_image(self, image):
        """Normalize the image to [0,1], add batch and channel dimensions, and convert to torch tensor."""
        if len(image.shape) == 2:  # Ensure the image is grayscale
            image = image[None, None, :, :]  # Add batch and channel dimensions: (1, 1, H, W)
        else:
            raise ValueError("Input image must be grayscale.")
        return torch.tensor(image / 255.0, dtype=torch.float32).to(self.device)

    def get_features(self, image):
        normalized_image = self.normalize_image(image)
        feats = self.extractor.extract(normalized_image)
        return feats
    

def set_neural_feature_extractor(device=None):
    """
    Returns the correct feature extractor based on the user's choice.
    Supported options: 1 for Superpoint. 
    """
    return SuperPointFeatureExtractor(device)