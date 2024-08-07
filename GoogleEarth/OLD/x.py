import cv2 # this module is used for image processing
import torch # this module is used for deep learning
import os # this module is used for file operations
import numpy as np # this module is used for numerical operations
from lightglue import LightGlue, SuperPoint # this module is used for feature extraction

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # this line checks if a GPU is available

# SuperPoint with custom parameters
extractor = SuperPoint(
    max_num_keypoints=10,  # more keypoints imply more features but also more computation and potentially more noise
    detection_threshold=0.1,  # lower threshold implies more keypoints but also more noise
    nms_radius=15  # a smaller radius implies more keypoints but also more noise. computation increases with smaller radius
).eval().to(device)

def normalize_image(image):
    """Normalize the image to [0,1] and convert to torch tensor."""
    return torch.tensor(image / 255.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0) # this line normalizes the image by dividing by 255 and converts it to a tensor. The unsqueeze function adds a batch dimension. torch expects a batch dimension in the input.. a tensor of shape (1, 1, H, W) is expected. a tensor is a multi-dimensional array. H is the height of the image and W is the width of the image. The first 1 is the batch size and the second 1 is the number of channels. The image is grayscale so it has one channel. batch size is 1 because we are processing one image at a time. torch is used for deep learning and it expects input in this format.

def detect_and_compute_superpoint(image):
    """Detect and compute SuperPoint features."""
    image = normalize_image(image) # this line normalizes the image
    feats = extractor.extract(image) # this line extracts features from the image
    return feats # this line returns the features. features are keypoints and descriptors. keypoints are points of interest in the image. descriptors are vectors that describe the keypoints. descriptors are used to match keypoints between images. keypoints are used to visualize the points of interest. the dimensions of the keypoints and descriptors are (1, N, 2) and (1, N, 256) respectively. N is the number of keypoints. 2 is the number of coordinates in a keypoint. 256 is the length of a descriptor vector.

def draw_keypoints(image, keypoints):
    """Draw keypoints on the image."""
    keypoints = keypoints.reshape(-1, 2) # Reshape keypoints to (N, 2)
    for keypoint in keypoints: # this line iterates over the keypoints
        x, y = keypoint[0], keypoint[1] # this line unpacks the keypoint into x and y coordinates
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), 1) # this line draws a circle at the keypoint location. the circle has a radius of 2 pixels. the color of the circle is green. the thickness of the circle is 1 pixel.
    return image

def main():
    # Directory containing the images
    directory = './GoogleEarth/SET1' # this line specifies the directory containing the images

    # Iterate over images and print features
    for i in range(1, 5):
        image_path = os.path.join(directory, f'{i}.jpg') # this line constructs the path to the image
         
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # this line reads the image in grayscale
        features = detect_and_compute_superpoint(image) # this line detects and computes the features using SuperPoint
        
        keypoints = features['keypoints'].cpu().numpy() # this line converts the keypoints from torch tensor to numpy array
        print(f"Image {i} Keypoints:\n{keypoints}\n") # Print the keypoints for debugging
        
        image_with_keypoints = draw_keypoints(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), keypoints) # this line converts the grayscale image to BGR color and draws the keypoints

        # Display the image with keypoints
        cv2.imshow(f"Image {i} - Keypoints", image_with_keypoints) # this line displays the image with keypoints
        cv2.waitKey(0)  # Wait for a key press to move to the next image
        cv2.destroyAllWindows() # this line closes the window

if __name__ == "__main__":
    main() # this line runs the main function
