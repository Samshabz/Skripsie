import torch
from lightglue import SuperPoint

# SuperPoint Feature Extractor
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

class SuperPointFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, device):
        self.device = device
        self.extractor = SuperPoint(
            max_num_keypoints=1024, #2048,
            detection_threshold=0.000000015,# lower implies more keypoints - prior: 0000015
            nms_radius=5
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
    

def set_neural_feature_extractor(detector_choice, device=None):
    """
    Returns the correct feature extractor based on the user's choice.
    Supported options: 1 for Superpoint. 
    """
    if detector_choice == 1:
        return SuperPointFeatureExtractor(device)
    else:
        raise ValueError(f"Invalid detector choice: {detector_choice}")