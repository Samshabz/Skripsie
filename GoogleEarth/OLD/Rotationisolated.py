import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations

class UAVRotationAnalyzer:
    def __init__(self):
        # Initialize the SIFT detector
        self.sift = cv2.SIFT_create()

        # Set up FLANN-based matcher parameters for SIFT (floating-point features)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)  # Increase the number of checks
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def crop_image(self, image):
        """Crop the top and bottom 10% of the image."""
        height = image.shape[0]
        cropped_image = image[int(height * 0.1):int(height * 0.9), :]
        return cropped_image

    def calculate_rotation_change(self, image1, image2, severity):
        """Calculate rotation change between two images using SVD."""
        # Crop and convert images to grayscale
        gray1 = cv2.cvtColor(self.crop_image(image1), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.crop_image(image2), cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors
        keypoints1, descriptors1 = self.sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(gray2, None)

        if descriptors1 is None or descriptors2 is None:
            print("Warning: No descriptors found in one or both images. Skipping.")
            return None

        # Match descriptors using FLANN-based matcher
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            print("Warning: Not enough good matches found. Skipping.")
            return None

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Use SVD to estimate the rotation angle between src_pts and dst_pts
        center_src = np.mean(src_pts, axis=0)
        center_dst = np.mean(dst_pts, axis=0)

        src_pts_centered = src_pts - center_src
        dst_pts_centered = dst_pts - center_dst

        H = np.dot(src_pts_centered.T, dst_pts_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(U, Vt)

        # Extract the rotation angle from the rotation matrix
        theta = np.arctan2(R[1, 0], R[0, 0])
        rotation_angle = np.degrees(theta)

        return rotation_angle

def main():
    directory = './GoogleEarth/DATASETS/DATSETAMAZ'  # Directory containing images
    image_range = range(1, 15)  # Define the range of images to process
    rotation_analyzer = UAVRotationAnalyzer()
    angles = []

    for i in range(min(image_range), max(image_range)):
        image1_path = os.path.join(directory, f'{i}.jpg')
        image2_path = os.path.join(directory, f'{i+1}.jpg')

        image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
        image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

        if image1 is None or image2 is None:
            print(f"Error: Could not load images {i} or {i+1}. Skipping.")
            continue

        rotation_change = rotation_analyzer.calculate_rotation_change(image1, image2, severity=5)

        if rotation_change is not None:
            angles.append(rotation_change)

    print("angles", angles)
    print("max is ", np.max(angles))
    print("mean is ", np.mean(angles))

if __name__ == "__main__":
    main()










