import cv2  # This module is used for image processing
import numpy as np  # This module is used for numerical operations
import os  # This module is used for file operations
import torch  # This module is used for deep learning
from lightglue import LightGlue, SuperPoint  # These modules are used for feature extraction
from lightglue.utils import rbd  # Utility function for removing batch dimension
import matplotlib.pyplot as plt  # This module is used for plotting

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if a GPU is available

# SuperPoint+LightGlue with custom parameters
extractor = SuperPoint(
    max_num_keypoints=50000,  # Maximum number of keypoints. More keypoints may improve matching but increase computation. It can also increase the chance of false matches / noise. 
    detection_threshold=0.003,  # Detection threshold. Lower values may detect more keypoints but increase noise.
    nms_radius=15  # Non-maximum suppression radius. Higher values may reduce the number of keypoints and improve matching (by reducing noisy duplicate points).
).eval().to(device) # Set the model to evaluation mode and move it to the device

matcher = LightGlue( # Initialize LightGlue
    features='superpoint', # Use SuperPoint features
    filter_threshold=0.1  # Custom filter threshold. Lower values may increase the number of matches but may also increase noise.
).eval().to(device)

def normalize_image(image):
    """Normalize the image to [0,1] and convert to torch tensor."""
    return torch.tensor(image / 255.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)  # Normalize the image and convert to tensor. image/255.0 normalizes the image to [0,1]. dtype=torch.float32 converts the image to float32. unsqueeze(0) adds a batch dimension. unsqueeze(0) adds a channel dimension. the final shape is [1, 1, H, W]. The first 1 is the batch dimension, the second 1 is the channel dimension, and H, W are the height and width of the image. The reason for adding the batch and channel dimensions is that the model expects input in this format. The model uses these for batch processing and to support multi-channel images. The return type is a torch tensor.

def crop_image(image):
    """Crop the top and bottom 15% of the image."""
    height = image.shape[0] # Get the height of the image. image.shape returns the height, width, and number of channels of the image. The height is the first element. 
    cropped_image = image[int(height * 0.25):int(height * 0.75), :]  # Crop the image. int(height * 0.15) calculates 15% of the height. int(height * 0.85) calculates 85% of the height. The colon : selects all columns. The cropped image contains the central 70% of the image. The return type is a numpy array.
    return cropped_image # Return the cropped image of type numpy array.

def detect_and_compute_superpoint(image):
    """Detect and compute SuperPoint features."""
    cropped_image = crop_image(image)  # Crop the image
    normalized_image = normalize_image(cropped_image)  # Normalize the cropped image
    feats = extractor.extract(normalized_image)  # Extract features. extractor is a SuperPoint model. extract is a method that extracts keypoints and descriptors from an image. The input is a normalized image tensor. The output is a dictionary containing keypoints, descriptors, and scores. The keypoints are the detected keypoints. The descriptors are the feature descriptors. The scores are the detection scores. The return type is a dictionary. Each feature has a key-value pair. The keys are 'keypoints', 'descriptors', and 'scores'. The values are the corresponding tensors. The keypoints tensor has shape [1, N, 2]. The N is the number of keypoints. The 2 is the x, y coordinates of the keypoints. The descriptors tensor has shape [1, N, 256]. The 256 is the dimension of the descriptor. The scores tensor has shape [1, N]. The return type is a dictionary. So feats has the following structure: {'keypoints': tensor([[[x1, y1], [x2, y2], ..., [xN, yN]]]), 'descriptors': tensor([[[d1], [d2], ..., [d256]]]), 'scores': tensor([[s1, s2, ..., sN]])}. keypoints are a pair of x, y coordinates. descriptors are the feature descriptors. scores are the detection scores. The return type is a dictionary. There is a score, descriptor, and keypoint for each feature. The return type is a dictionary. A dictionary is: {'key': value}. The key is the name of the feature. The value is the feature tensor. The return type is a dictionary. 
    return feats

def match_features_superpoint(featsA, featsB):
    """Match features using LightGlue."""
    matches = matcher({'image0': featsA, 'image1': featsB})  # Match features. Matcher is a lightglue model. it compares the descriptors of a pair of images and returns the matches. The input is a dictionary containing the descriptors of the two images. The output is a dictionary containing the matches. The matches are the indices of the matched features. The return type is a dictionary. The input is a dictionary. The keys are 'image0' and 'image1'. The values are the feature dictionaries of the two images. The feature dictionary contains the keypoints, descriptors, and scores. The return type is a dictionary. The output is a dictionary. The keys are 'matches'. The value is
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]  # Remove batch dimension. rbd(x) removes the batch dimension from the tensor x. The input is a tensor. The output is a tensor without the batch dimension. The tensor is now of dimension [N, ...]. featsA is the feature dictionary of image A. featsB is the feature dictionary of image B. matches are the matched indices. The return type is a tuple. The tuple contains the feature dictionaries of image A and B, and the matched indices. The feature dictionaries are of type numpy array. The matched indices are of type numpy array. The return type is a tuple. The tuple contains the feature dictionaries of image A and B, and the matched indices. The feature dictionaries are of type numpy array. The matched indices are of type numpy array. The reason we remove the dimension corresponding to the batch is that the model expects input without the batch dimension. The model uses the batch dimension for batch processing. 
    return featsA, featsB, matches['matches']

def custom_nms(keypoints, radius): # in simple, it removes keypoints that are too close to each other
    """Custom Non-Maximum Suppression (NMS) to control keypoint distribution."""
    if len(keypoints) == 0: # If there are no keypoints,
        return keypoints # Return the keypoints
    keypoints = sorted(keypoints, key=lambda x: -x.response)  # The array has each value made negative, then sort using the default ascending order, then make each value negative again. This is equivalent to sorting in descending order. The key argument specifies a function that returns a key to use for sorting. The lambda function returns the negative response of the keypoint. The response is a measure of the keypoint strength. The keypoint with the highest response is first. The return type is a list. A list vs a dictionary: A list is an ordered collection of items. A dictionary is an unordered collection of items. The return type is a list. 
    keep = [] 
    for kp in keypoints: #iterate through the list of keypoints
        if all(np.linalg.norm(np.array(kp.pt) - np.array(keep_kp.pt)) >= radius for keep_kp in keep): # iterate through a new list keep. The keypoint is compared against whatever is in the keep list and will only be added to the keep list if its not close to any element therein. For isntance, the first (strongest) keypoint is autoadded as keep is empty, thereafter the second strongest is compared against the keep list (with only the one strongest item therein).  
            keep.append(kp)  # Keep keypoints that are not within the NMS radius
    return keep # Return the keypoints which are now filtered (distance) and sorted. 

class UAVNavigator: # UAVNavigator class. A class is defined as clearly put, a blueprint for creating objects. By blueprint I mean that it defines the attributes and methods that an object will have. Attributes are variables that store data. Methods are functions that perform actions. So, a class contains data variables and functions that operate on the data. The object is an instance of the class. 
    def __init__(self, gps_per_pixel_x, gps_per_pixel_y): # Constructor method. The constructor (__init__) is a special method that is called when an object is created. It initializes the object. Self implies that the variables will only be related to the single object (the func that calls it) and not change the entire class. E.g. changing the parameters of a dog named buddy as opposed to changing the parameters of all dogs. We pass in the GPS conversion factors for longitude and latitude.
        self.gps_per_pixel_x = gps_per_pixel_x  # Store GPS conversion factor for longitude. This does not actually have to be an instance variable, but it is stored as one.
        self.gps_per_pixel_y = gps_per_pixel_y  # Store GPS conversion factor for latitude. This does not actually have to be an instance variable, but it is stored as one.
        
        # Lists to store images, descriptors, keypoints, and GPS coordinates
        self.stored_images = []
        self.stored_features = []
        self.stored_gps = []

    def add_image(self, image, gps_coordinates): # Method to add an image and its GPS coordinates to the stored list. We need to keep track of the images, features, and GPS coordinates for reverse navigation. Well, not necessarily reverse the images. HAVE A LOOK HERE. 
        """Add an image and its GPS coordinates to the stored list."""
        features = detect_and_compute_superpoint(image)  # Detect and compute features
        self.stored_images.append(image)  # Store the original image
        self.stored_features.append(features)  # Store the features
        self.stored_gps.append(gps_coordinates)  # Store the GPS coordinates

    def _compute_homography(self, feats1, feats2):
        """Compute the homography matrix using RANSAC from matched features."""
        feats1, feats2, matches = match_features_superpoint(feats1, feats2)  # Match features
        
        keypoints1 = feats1['keypoints'].cpu().numpy()  # Convert keypoints to numpy array. The .cpu method moves the tensor to the CPU. The .numpy method converts the tensor to a numpy array. The keypoints tensor is a tensor of shape [1, N, 2]. The N is the number of keypoints. The 2 is the x, y coordinates of the keypoints. The return type is a numpy array.
        keypoints2 = feats2['keypoints'].cpu().numpy()  # Convert keypoints to numpy array
        matches = matches.cpu().numpy()  # Convert matches to numpy array using the cpu. 

        if len(matches) < 4:
            return None # Return None if there are less than 4 matches

        src_pts = keypoints1[matches[:, 0]].reshape(-1, 1, 2)  # Source points for homography. Source points are the keypoints from the first image. The matches[:, 0] selects the indices of the matched keypoints from the first image. The keypoints are reshaped to a 3D array of shape [N, 1, 2]. The N is the number of keypoints. The 1 is the number of points. The 2 is the x, y coordinates of the keypoints. The return type is a numpy array. Its in this format to be compatible with the cv2.findHomography function.
        dst_pts = keypoints2[matches[:, 1]].reshape(-1, 1, 2)  # Destination points for homography. We should keep track of which image is called as the source and destination. 
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)  # Compute homography matrix. M is the homography matrix. mask is the mask of inliers ( this is an array which is 1 for inliers and 0 for outliers - RANSAC). The src_pts are the source points. The dst_pts are the destination points. The cv2.RANSAC flag specifies the RANSAC method. The 5.0 is the RANSAC threshold (1-10 typical). A higher threshold allows more outliers. The return type is a tuple. The tuple contains the homography matrix and the mask. The homography matrix is a 3x3 matrix that allows us to figure out the transformation between the two images. More specifically, we can find the translation and rotation which allows us to then take the centre of one image (known) and find the centre of the other image (unknown).The return type is a tuple.
        return M, mask, matches

    def infer_current_gps(self, current_image, current_index):
        """Infer the current GPS location based on the highest correlated stored image."""
        current_features = detect_and_compute_superpoint(current_image)  # Detect and compute features

        max_matches = 0
        best_homography = None
        best_index = -1

        # Only compare with prior images
        for i in range(current_index): # We only compare with prior images. We do not compare with future images.
            result = self._compute_homography(current_features, self.stored_features[i]) # returns a homography matrix, mask, and matches. We use this to find the most matches, and subsequently the homography thereof. That is, the translation and rotation between the two images.
            if result is None:
                continue

            M, mask, matches = result  # note that mask is more the amount of inliers whereas matches is the actual matches.
            good_matches = np.sum(mask)  # Count the number of good matches

            if good_matches > max_matches:
                max_matches = good_matches
                best_homography = M
                best_index = i

        if best_homography is not None and best_index != -1:
            h, w = self.stored_images[best_index].shape[:2] # Get the height and width of the image. We return the first two elements of the shape of the image (H, W, NOT the number of channels).
            center_pt = np.array([[w / 2, h / 2]], dtype='float32').reshape(-1, 1, 2)  # Center point of the image. reshape it to 1 (number of points), 1 (number of dimensions), 2 (x, y coordinates). The center point is the middle of the image. The x coordinate is half the width. The y coordinate is half the height. The return type is a numpy array.
            transformed_center = cv2.perspectiveTransform(center_pt, best_homography)  # Transform center point
            delta_x, delta_y = transformed_center[0][0] - center_pt[0][0]  # Calculate the shift in x and y

            gps_x, gps_y = self.stored_gps[best_index]
            current_gps_x = gps_x + delta_x * self.gps_per_pixel_x  # Calculate the current GPS longitude
            current_gps_y = gps_y + delta_y * self.gps_per_pixel_y  # Calculate the current GPS latitude

            return best_index, (current_gps_x, current_gps_y)
        return None, None

def parse_gps(file_path):
    """Parse GPS coordinates from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file: # Open the file in read mode. The encoding is utf-8. The file is opened in a context manager. The context manager automatically closes the file after the block of code is executed.
        lines = file.readlines()  # Read all lines from the file. The readlines method reads all lines from the file and returns a list of lines. Each line is a string. The return type is a list of strings.
        lat_str = lines[0].strip() # Latitude string. The strip method removes leading and trailing whitespaces. The latitude is the first line in the file. The return type is a string.
        lon_str = lines[1].strip()
        
        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_str)  # Parse latitude using DMS
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_str)  # Parse longitude

        lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)  # Convert latitude to decimal
        lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)  # Convert longitude to decimal

        return lat, lon

def parse_dms(dms_str):
    """Parse degrees, minutes, seconds from a DMS string."""
    dms_str = dms_str.replace('°', ' ').replace('\'', ' ').replace('"', ' ')  # Replace degree, minute, second symbols with spaces to split. Basically, we replace the symbols with spaces so that we can split the string.
    parts = dms_str.split() # Split the string into parts. The split method splits the string into a list of strings. The default separator is whitespace. The return type is a list of strings.
    deg = int(parts[0])  # Degrees
    min = int(parts[1])  # Minutes
    sec = float(parts[2])  # Seconds
    dir = parts[3]  # Direction (N, S, E, W)
    return deg, min, sec, dir

def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0  # Convert to decimal
    if dir in ['S', 'W']:
        decimal = -decimal  # Negative for South and West
    return decimal
    
def main():
    # the amount of metres per pixel is 1092 metres / 596 pixels = 1.8322 metres per pixel. This is the conversion factor. Then, to get this to GPS per pixel we times by degrees per metre. This is 1/111139. This is the conversion factor. 
    gps_per_pixel_x = 1.6486e-5  # Longitude
    gps_per_pixel_y = 1.6486e-5  # Latitude. This is a bit of a problem (calculation). Also not sure why they work better being different.
    navigator = UAVNavigator(gps_per_pixel_x, gps_per_pixel_y)  # Initialize the UAVNavigator

    # Directory containing the images and GPS files
    directory = './GoogleEarth/SET1' 

    # Lists to store actual and estimated GPS coordinates
    actual_gps_list = [] # We store the actual GPS coordinates in a list.
    estimated_gps_list = [] # We store the estimated GPS coordinates in a list.
    num_images = 10
    # Add images and GPS coordinates to the navigator
    for i in range(1, num_images+1): # start from 1 up until 1+pics
        image_path = os.path.join(directory, f'{i}.jpg')    # Image path
        gps_path = os.path.join(directory, f'{i}.txt') # GPS path
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale (we should change this to RGB perhaps)
        gps_coordinates = parse_gps(gps_path)  # Parse the GPS coordinates. Parse means to analyze a string and convert it into a different data type. The input is a string. The output is a tuple of latitude and longitude. 
        
        navigator.add_image(image, gps_coordinates)  # Add the image and GPS coordinates to the navigator

    # Simulate backward flight (GPS lost)
    for i in range(num_images-1, 0, -1): # The first argument is the start value. The second argument is the end value. The third argument is the step value. 
        current_image = navigator.stored_images[i] # Get the current image
        best_index, estimated_gps = navigator.infer_current_gps(current_image, i) # Infer the current GPS location

        if best_index is not None and estimated_gps is not None:
            actual_gps = navigator.stored_gps[i] # Get the actual GPS location. Navigator is the UAVNavigator object. stored_gps is a list of GPS coordinates. The index i selects the GPS coordinates for the current image. The return type is a tuple. The tuple contains the latitude and longitude.
            actual_gps_list.append(actual_gps)
            estimated_gps_list.append(estimated_gps)
            print(f"Image {i+1} best match with the following: Image {best_index+1}") # the +1 is to make it human readable (1-indexed)
            print(f"Estimated GPS Location for image {i+1}: {estimated_gps}")
            print(f"Actual GPS Location for image {i+1}: {actual_gps}")
            print(f"Deviation-x (%) = {abs(estimated_gps[0] - actual_gps[0]) / abs(actual_gps[0]) * 100} at image {i+1} based off matched image (assumed) {i}") 
            print(f"Deviation-y (%) = {abs(estimated_gps[1] - actual_gps[1]) / abs(actual_gps[1]) * 100} at image {i+1} based off matched image (assumed) {i}")   
            # now lets print the deviation in metres
            print(f"Deviation-x (m) = {abs(estimated_gps[0] - actual_gps[0]) * 111139}")
            print(f"Deviation-y (m) = {abs(estimated_gps[1] - actual_gps[1]) * 111139}")
        else:
            print(f"Image {i+1}: Unable to estimate GPS location")

    # Separate the GPS coordinates into latitude and longitude
    actual_gps_x = [coord[0] for coord in actual_gps_list] # Extract the longitude from the actual GPS coordinates. The actual GPS coordinates are a list of tuples. The longitude is the first element of the tuple. The list comprehension extracts the longitude from each tuple. The return type is a list.
    actual_gps_y = [coord[1] for coord in actual_gps_list]
    estimated_gps_x = [coord[0] for coord in estimated_gps_list]
    estimated_gps_y = [coord[1] for coord in estimated_gps_list]

    # Plot the actual and estimated GPS coordinates for longitude
    plt.figure()
    plt.plot(range(1, len(actual_gps_x)+1), actual_gps_x, label='Actual Longitude') # The range function generates a sequence of numbers. The start is 1. The end is the length of the actual GPS coordinates plus 1. The step is 1. The actual GPS coordinates are a list of longitude values. The plot function plots the actual longitude values. The label is 'Actual Longitude'. The return type is a plot object.
    plt.plot(range(1, len(estimated_gps_x)+1), estimated_gps_x, label='Estimated Longitude')
    # the same thing. From range 1 -> len(images) + 1. The plot function plots the estimated longitude values. The label is 'Estimated Longitude'. The return type is a plot object.
    plt.xlabel('Image Index')
    plt.ylabel('Longitude')
    plt.legend()
    plt.title('Actual vs Estimated GPS Longitude')
    

    # Plot the actual and estimated GPS coordinates for latitude
    plt.figure()
    plt.plot(range(1, len(actual_gps_y)+1), actual_gps_y, label='Actual Latitude')
    plt.plot(range(1, len(estimated_gps_y)+1), estimated_gps_y, label='Estimated Latitude')
    plt.xlabel('Image Index')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('Actual vs Estimated GPS Latitude')
    plt.show()

if __name__ == "__main__": # This only runs if the script is run directly, not if it is imported as a module. This is because the __name__ variable is set to "__main__" iff the script is run directly. Useful if this code was being imported into another script. Else if we didnt care wed simply call main() at the end of the script.
    main()  
