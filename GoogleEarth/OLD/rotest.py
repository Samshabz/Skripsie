import cv2
import numpy as np
import os

# Directory containing images
image_directory = './GoogleEarth/DATASETS/DATSETROT/'

# Function to crop top and bottom 10% of the image
def crop_image(img):
    height, width = img.shape[:2]
    crop_size = int(height * 0.05)
    return img[crop_size:height-crop_size, :]  

# Function to estimate affine rotation with more accurate settings
def estimate_affine_rotation(image1_gray, image2_gray):
    """Estimate the rotation angle between two images using a more accurate affine transformation with AKAZE."""
    # Detect keypoints and descriptors using AKAZE
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(image1_gray, None)
    kp2, des2 = akaze.detectAndCompute(image2_gray, None)

    # BFMatcher for matching keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Use crossCheck for more accurate matches
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter top matches (limit to 400 max for performance)
    top_matches = matches[:min(len(matches), 400)]

    # Only return if we have enough matches
    if len(top_matches) < 250:
        return None, len(top_matches)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

    # Estimate affine transformation using RANSAC
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    if M is None:
        return None, len(top_matches)

    # Extract rotation angle from the affine matrix
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))  # Convert radians to degrees
    return angle, len(top_matches)

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

# Recursive function to calculate the global heading
def calculate_global_heading(images, index, current_heading):
    """Recursively calculate global heading for image at index `index` based on previous images."""
    if index == 0:
        print(f"Image: {images[index][0]}, Global Heading: 0 degrees (Reference)")
        return 0.0  # Reference image, heading is 0 degrees
    
    # Get the previous image heading recursively
    previous_heading = calculate_global_heading(images, index - 1, current_heading)

    # Estimate relative rotation between current and previous image
    relative_rotation, num_matches = estimate_affine_rotation(images[index - 1][1], images[index][1])

    if relative_rotation is None:
        print(f"Image: {images[index][0]}, Insufficient matches for accurate estimation.")
        return previous_heading  # Return previous heading if estimation fails
    
    # Update global heading by adding relative rotation
    global_heading = previous_heading + relative_rotation

    print(f"Image: {images[index][0]}, Relative Rotation: {relative_rotation} degrees, Global Heading: {global_heading} degrees")
    
    return global_heading

# Main function to calculate the headings
def calculate_headings():
    images = load_images(image_directory)

    if len(images) < 2:
        print("Not enough images for comparison.")
        return
    
    # Calculate the global heading for each image recursively
    for i in range(len(images)):
        calculate_global_heading(images, i, 0)

if __name__ == "__main__":
    calculate_headings()
