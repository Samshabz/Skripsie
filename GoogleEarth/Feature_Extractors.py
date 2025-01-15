
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
        self.detector = cv2.ORB_create(nfeatures=nfeatures) if nfeatures == -1 else cv2.ORB_create(
            nfeatures=nfeatures,            # Capture more keypoints
            scaleFactor=1.1,           # More scale layers. 1.3 gives an error. 
            nlevels=64,                # Increase pyramid levels for different scales 64 abv 32
            edgeThreshold=15,          # Lower to detect near image borders 15 above 5
            WTA_K=2,                   # Faster keypoint detection. 4 is worse than 2
            scoreType=cv2.ORB_HARRIS_SCORE  ,  # Lenient keypoint scoring # fast is slower but more accurate than harris
            patchSize=21,              # Smaller patch size to capture fine features. 21 better than 11 by alot
            fastThreshold=4           # Lower sensitivity for lenient detection. This and scale factor are big factors in the number of keypoints. 0, 1 is extremely slow not usable. 5 is fine. 3 has lots of matches but super slow. 4 seems decent. 
        )

    def get_keydes(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        return keypoints, descriptors

    def get_features(self, image):
        keypoints, _ = self.detector.detectAndCompute(image, None)
        return keypoints


class SIFTdetector(BaseFeatureExtractor):
    def __init__(self):
        self.detector = cv2.SIFT_create(
            nfeatures=3000,            # Capture more keypoints
            nOctaveLayers=4,           # Increase pyramid layers for different scales
            contrastThreshold=0.04,    # Lower contrast threshold for more keypoints
            edgeThreshold=10,          # Lower to detect near image borders
            sigma=1.6                  # Standard deviation for Gaussian blur
        )


    def get_keydes(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def get_features(self, image):
        keypoints, _ = self.detector.detectAndCompute(image, None)
        return keypoints


# AKAZE Feature Extractor
class AKAZEFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, threshold=0.0001):
        
        self.detector = cv2.AKAZE_create(
            nOctaves=16,  
            nOctaveLayers=6, 
            descriptor_size=128,
            threshold=threshold

        )
        # self.detector.setThreshold(threshold)
    # if threshold == 0.0001 else cv2.AKAZE_create(
    # descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,  # Use MLDB descriptors for more accuracy
    # descriptor_size=256,                        # Larger descriptor for more detail. 256 is better than 128. 512 doesnt run
    # descriptor_channels=3,                      # Standard channels. 5 doesnt run
    # ,                           # Lower threshold for more keypoints
    # nOctaves=16,                                 # More octaves for scale robustness. 16 worse than 8
    # nOctaveLayers=6,                            # More layers per octave for accurate multi-scale detection
    # diffusivity=cv2.KAZE_DIFF_CHARBONNIER        # Most accurate diffusion process
# )
                      
        
        # self.detector.setThreshold(threshold)
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
    elif detector_choice == 4:
        return SIFTdetector()


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
            # max_num_keypoints=1024, #2048,
            # detection_threshold=0.0000015,# lower implies more keypoints - prior: 0000015
            # nms_radius=1 # lower implies more keypoints

            nms_radius= 4,  # Increase to suppress more keypoints
            max_num_keypoints = 1700,  # Limit the number of keypoints to extract
            detection_threshold =0.0000015,  # Increase threshold for more robust keypoints
            # remove_borders =1,  # Increase border distance to filter more edge keypoints
            # descriptor_dim =128,  # Cannot change, it was trained with 256
            # channels =[8, 8, 16, 16, 32],  # Reduce channels for faster processing. 127 seconds / 29 mean error (with channel reduction on) vs runtime of 116 with off. but 31 error with off. 


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