import cv2
import numpy as np

# Function to crop top and bottom 10% of the image
def crop_image(img):
    height, width = img.shape[:2]
    crop_size = int(height * 0.1)  # 10% of the height
    return img[crop_size:height-crop_size, :]  # Crop top and bottom

# Load and crop images
image1_path = './GoogleEarth/SET2/7.jpg'
image2_path = './GoogleEarth/SET2/6.jpg'

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

if image1 is None or image2 is None:
    print("Error: Could not load one or both images.")
    exit()

# Crop the images
image1_cropped = crop_image(image1)
image2_cropped = crop_image(image2)

# Convert to grayscale for translation methods
image1_gray = cv2.cvtColor(image1_cropped, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2_cropped, cv2.COLOR_BGR2GRAY)

def estimate_translation_phase_correlation(image1_gray, image2_gray):
    """Estimate translation using Phase Correlation."""
    shift, response = cv2.phaseCorrelate(np.float32(image1_gray), np.float32(image2_gray))
    print(f"Phase Correlation estimated translation: {shift}")
    return shift

def estimate_translation_optical_flow(image1_gray, image2_gray):
    """Estimate translation using Optical Flow."""
    # Detect good features to track using Shi-Tomasi corner detector
    src_pts = cv2.goodFeaturesToTrack(image1_gray, maxCorners=1000000, qualityLevel=0.6, minDistance=0.00001)
    
    if src_pts is None:
        print("No good features to track found.")
        return None
    
    # Calculate optical flow using Lucas-Kanade method
    dst_pts, status, err = cv2.calcOpticalFlowPyrLK(image1_gray, image2_gray, src_pts, None)
    
    # Filter valid points
    valid_pts = np.where(status == 1)
    
    if len(valid_pts[0]) < 4:
        print("Not enough valid points found after optical flow.")
        return None

    # Calculate average translation between points
    translation = np.mean(dst_pts[valid_pts] - src_pts[valid_pts], axis=0).ravel()
    print(f"Optical Flow estimated translation: {translation}")
    return translation

def estimate_translation_keypoint_affine(image1_gray, image2_gray):
    """Estimate translation using Keypoints and Affine transformation."""
    # ORB Detector for keypoints
    AKAZE = cv2.AKAZE_create()

    # Detect keypoints and descriptors
    kp1, des1 = AKAZE.detectAndCompute(image1_gray, None)
    kp2, des2 = AKAZE.detectAndCompute(image2_gray, None)

    # BFMatcher for matching keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Extract locations of matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate affine transformation using RANSAC
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if M is None:
        print("Affine estimation failed.")
        return None

    # Extract translation
    translation_x = -M[0, 2]
    translation_y = M[1, 2]
    print(f"Keypoint Matching and Affine estimated translation: [{translation_x}, {translation_y}]")
    return translation_x, translation_y

# Main logic to choose method
def main():
    print("Choose a method for translation estimation:")
    print("1. Phase Correlation")
    print("2. Optical Flow")
    print("3. Keypoint Matching with Affine")

    choice = input("Enter 1, 2, or 3: ")

    if choice == '1':
        estimate_translation_phase_correlation(image1_gray, image2_gray)
    elif choice == '2':
        estimate_translation_optical_flow(image1_gray, image2_gray)
    elif choice == '3':
        estimate_translation_keypoint_affine(image1_gray, image2_gray)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
