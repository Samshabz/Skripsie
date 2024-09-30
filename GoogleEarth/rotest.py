import cv2
import numpy as np
import os

# Directory containing images
image_directory = './GoogleEarth/DATASETS/DATSETROT/'

# Function to crop top and bottom 10% of the image
def crop_image(img):
    height, width = img.shape[:2]
    crop_size = int(height * 0.1)  # 10% of the height
    return img[crop_size:height-crop_size, :]  # Crop top and bottom

# Function to estimate affine rotation
def estimate_affine_rotation(image1_gray, image2_gray):
    """Estimate the rotation angle between two images using an affine transformation."""
    # Detect keypoints and descriptors using AKAZE
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(image1_gray, None)
    kp2, des2 = akaze.detectAndCompute(image2_gray, None)

    # BFMatcher for matching keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate affine transformation using RANSAC
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if M is None:
        raise ValueError("Affine transformation estimation failed.")
    
    # Extract rotation angle from the affine matrix
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))  # Convert radians to degrees
    return angle

# Load and process all images
def load_images(directory):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
    images = []
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_file}")
            continue
        # Crop and convert to grayscale
        image_cropped = crop_image(image)
        image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
        images.append((image_file, image_gray))
    return images

# Main function to calculate the headings
def calculate_headings():
    images = load_images(image_directory)

    if len(images) < 2:
        print("Not enough images for comparison.")
        return
    
    # Initialize global heading with the first image (heading = 0 degrees)
    global_heading = 0
    reference_image = images[0][1]  # Gray image of the first image
    print(f"Image: {images[0][0]}, Global Heading: 0 degrees (Reference)")

    # Calculate the relative heading of all other images
    for i in range(1, len(images)):
        image_name, current_image_gray = images[i]
        previous_image_gray = images[i-1][1]  # Compare with the previous image

        try:
            # Calculate the relative rotation between the current and the previous image
            relative_rotation_angle = estimate_affine_rotation(previous_image_gray, current_image_gray)
            global_heading += relative_rotation_angle  # Accumulate the global heading

            print(f"Image: {image_name}, Relative Rotation: {relative_rotation_angle} degrees, Global Heading: {global_heading} degrees")

        except ValueError as e:
            print(f"Image: {image_name}, Error: {str(e)}")

if __name__ == "__main__":
    calculate_headings()
