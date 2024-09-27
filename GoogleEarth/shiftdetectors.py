import cv2
import numpy as np

def estimate_shift_from_points(image1, image2):
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) > 2 else image1
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) > 2 else image2

    # AKAZE Detector for keypoints
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

    # Extract translation (shift_x, shift_y)
    shift_x = M[0, 2]
    shift_y = M[1, 2]
    
    return shift_x, shift_y
