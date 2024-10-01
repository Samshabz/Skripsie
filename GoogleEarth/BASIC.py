import cv2
import numpy as np
from shiftdetectors import get_shifts

# Function to crop top and bottom 10% of the image
def crop_image(img):
    height, width = img.shape[:2]
    crop_size = int(height * 0.1)  # 10% of the height
    return img[crop_size:height-crop_size, :]  # Crop top and bottom

# Load and crop images # from 1->5, rotate -90, then move+x,-y - higher mag of y. 
anglesubj = -65
anglesubj2 = -43.5
rev_true = True
angletouse = anglesubj2

# false, 1 same as True 2
# ie using false and -65, -65 is angle of 6. so false is 7 first. ie u pass in the angle of what is second, which is the inference image -> that being rotated. 
# 6 -> -65
# 7 -> -43.5

if rev_true == True:
    angletouse = -90
    image1_path = './GoogleEarth/DATASETS/DATSETROT/1.jpg' # ref, ie i want to know how i move from ref to inf 1->5 (+x, -y)
    image2_path = './GoogleEarth/DATASETS/DATSETROT/5.jpg' # inf
else:
    angletouse = -0
    image1_path = './GoogleEarth/DATASETS/DATSETROT/5.jpg'
    image2_path = './GoogleEarth/DATASETS/DATSETROT/1.jpg'

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

def rotate_image(image, angle):
    """Rotate the image by a given angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

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
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC) # this is from the first image to the second image, in our case: inference to reference

    if M is None:
        raise ValueError("Affine transformation estimation failed.")
    
    # Extract rotation angle from the affine matrix
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))  # Convert radians to degrees
    print(f"Estimated rotation angle: {angle} degrees")
    
    return angle

def estimate_translation_phase_correlation(ref_img, inference_image, reference_heading, featsA=None, featsB=None, kp1=None, kp2=None, des1=None, des2=None):
    """Estimate translation using Phase Correlation with consistent global heading."""


    rotation_angle = -estimate_affine_rotation(inference_image, ref_img)
    rot_inf_img = rotate_image(inference_image, rotation_angle + reference_heading)
    rot_ref_img = rotate_image(ref_img, reference_heading)
    rotation_anglecheck = -estimate_affine_rotation(rot_inf_img, rot_ref_img)
    print(f"Rotation angle between images: {rotation_anglecheck}")
    return rot_inf_img, rot_ref_img
    # shift, response = cv2.phaseCorrelate(np.float32(rot_inf_img), np.float32(rot_ref_img))
    # translation_x, translation_y = shift

    translation_x, translation_y = get_shifts(None, None, kp1, kp2, des1, des2)

    return translation_x, translation_y




def estimate_translation_keypoint_affine(image1_gray, image2_gray):
    image2_gray = rotate_image(image2_gray, 15)
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
    # print("Choose a method for translation estimation:")
    # print("1. Phase Correlation")
    # print("2. Optical Flow")
    # print("3. Keypoint Matching with Affine")

    # choice = input("Enter 1, 2, or 3: ")
    choice='1'
    if choice == '1':
        estimate_translation_phase_correlation(image1_gray, image2_gray, angletouse, True)
    elif choice == '3':
        estimate_translation_keypoint_affine(image1_gray, image2_gray)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
# [184.64690108  29.12549606] 6-> 7 -65
#  [161.1241906   94.77218896] 6->7 -43.5
#  [-160.12220132  -96.97741915] 7->6 -65
# [-113.43816605 -148.91447768] 7->6 -43.5

# to ensure same system, must pass in the global heading of the reference object 
# new code? Do not pass in the heading of the reference, pass in that of the inference image 2
