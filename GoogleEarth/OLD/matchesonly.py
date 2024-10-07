import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations
import time  # To measure processing time

class UAVNavigator:
    def __init__(self, gps_to_pixel_scale, max_features=500, image_resize_factor=0.5):
        """Initialize the UAVNavigator with GPS to pixel scale."""
        self.gps_to_pixel_scale = gps_to_pixel_scale  # Pixels per meter
        self.stored_images = []
        self.stored_keypoints = []
        self.stored_descriptors = []
        self.stored_gps = []
        self.max_features = max_features
        self.image_resize_factor = image_resize_factor

        # Initialize the SIFT detector with a limited number of features
        self.sift = cv2.SIFT_create(nfeatures=max_features)

        # Set up FLANN-based matcher parameters for SIFT (floating-point features)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)  # Reduced checks for speed
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def clear_stored_data(self):
        """Clear stored images, keypoints, descriptors, and GPS data."""
        self.stored_images.clear()
        self.stored_keypoints.clear()
        self.stored_descriptors.clear()
        self.stored_gps.clear()

    def adjust_contrast(self, image):
        """Adjust the contrast of an image."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def sharpen_image(self, image):
        """Apply sharpening to an image."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def crop_image(self, image, kernel_size):
        """Crop the top and bottom 10% of the image and apply Gaussian blur."""
        height = image.shape[0]
        cropped_image = image[int(height * 0.1):int(height * 0.9), :]
        return cv2.GaussianBlur(cropped_image, (kernel_size, kernel_size), 0)

    def add_image(self, image, gps_coordinates, kernel_size):
        """Add an image and its GPS coordinates to the stored list."""
        # Reduce image size for faster processing
        image = cv2.resize(image, (0, 0), fx=self.image_resize_factor, fy=self.image_resize_factor)

        # Apply preprocessing steps
        image = self.adjust_contrast(image)
        image = self.sharpen_image(image)

        cropped_image = self.crop_image(image, kernel_size)
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

        if descriptors is None:
            print("Warning: No descriptors found for one image. Skipping.")
            return

        print(f"Image has {len(keypoints)} keypoints.")  # Print the number of keypoints

        self.stored_images.append(cropped_image)
        self.stored_keypoints.append(keypoints)
        self.stored_descriptors.append(descriptors)
        self.stored_gps.append(gps_coordinates)

    def find_best_matches(self, ratio_values):
        """Find and print the number of times each ratio leads to a consecutive best match."""
        for ratio in ratio_values:
            print(f"\nTesting with ratio: {ratio}")
            consecutive_matches_count = 0
            total_possible_consecutive = len(self.stored_images) - 1

            for i in reversed(range(1, len(self.stored_images))):
                max_custom_matches = 0
                best_index = -1

                for j in range(i):
                    start_time = time.time()
                    matches = self.matcher.knnMatch(self.stored_descriptors[i], self.stored_descriptors[j], k=2)

                    # Custom matches using the current ratio value
                    custom_matches = [m for m, n in matches if m.distance < ratio * n.distance]


                    if len(custom_matches) > max_custom_matches:
                        max_custom_matches = len(custom_matches)
                        best_index = j

                    print(f"Processed image {i} with {j} in {time.time() - start_time:.2f} seconds.")

                if best_index == i - 1:
                    consecutive_matches_count += 1
            
            print(f"Ratio {ratio} achieved {consecutive_matches_count}/{total_possible_consecutive} consecutive best matches.")

def parse_gps(file_path):
    """Parse GPS coordinates from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lat_str, lon_str = lines[0].strip(), lines[1].strip()
        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_str)
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_str)
        lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
        lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)
        return lat, lon

def parse_dms(dms_str):
    """Parse degrees, minutes, seconds from a DMS string."""
    dms_str = dms_str.replace('°', ' ').replace('\'', ' ').replace('"', ' ')
    parts = dms_str.split()
    deg, min, sec, dir = int(parts[0]), int(parts[1]), float(parts[2]), parts[3]
    return deg, min, sec, dir

def convert_to_decimal(deg, min, sec, dir):
    """Convert DMS to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0
    return -decimal if dir in ['S', 'W'] else decimal

def main():
    gps_to_pixel_scale = 596 / 1092  # Pixels per meter
    navigator = UAVNavigator(gps_to_pixel_scale, max_features=10000, image_resize_factor=0.95)
    directory = './GoogleEarth/SET1'
    num_images = 13
    kernels_to_test = [3]
    ratio_values = [0.75, 0.9, 0.95, 0.995, 0.99955]  # Different ratio values to test

    for kernel in kernels_to_test:
        navigator.clear_stored_data()  # Clear stored data before each kernel test
        for i in range(1, num_images + 1):
            image_path = os.path.join(directory, f'{i}.jpg')
            gps_path = os.path.join(directory, f'{i}.txt')
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
            gps_coordinates = parse_gps(gps_path)
            navigator.add_image(image, gps_coordinates, kernel)

        navigator.find_best_matches(ratio_values)

if __name__ == "__main__":
    main()
