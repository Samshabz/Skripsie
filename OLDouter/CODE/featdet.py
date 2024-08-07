import cv2
import numpy as np

# Parameters to control feature detection
nfeatures = 500  # Number of best features to retain
contrastThreshold = 0.08  # Increase to reduce number of features
edgeThreshold = 15  # Increase to filter out more edge-like features
sigma = 1.6  # Default value for SIFT detector

def enhance_contrast_clahe(image, clip_limit=4.0, tile_grid_size=(8, 8)):
    print("Enhancing contrast using CLAHE...")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def detect_features_sift(image, nfeatures, contrastThreshold, edgeThreshold, sigma):
    print("Detecting features using SIFT...")
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    print(f"Detected {len(keypoints)} keypoints.")
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def calculate_mean_change(matches, keypoints1, keypoints2):
    distances = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
        distances.append(distance)
    mean_change = np.mean(distances)
    return mean_change

# Load the images
image_path1 = 'image1.jpg'  # Path to your first image
image_path2 = 'image2.jpg'  # Path to your second image

print(f"Loading images from {image_path1} and {image_path2}...")
img1 = cv2.imread(image_path1, 0)  # Load first image in grayscale
img2 = cv2.imread(image_path2, 0)  # Load second image in grayscale

if img1 is None or img2 is None:
    print(f"Error: Unable to load images at {image_path1} and {image_path2}")
else:
    # Resize images to a more manageable size if they are too large
    max_dimension = 800
    if max(img1.shape) > max_dimension:
        scale = max_dimension / max(img1.shape)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Resized image1 to {img1.shape}")

    if max(img2.shape) > max_dimension:
        scale = max_dimension / max(img2.shape)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Resized image2 to {img2.shape}")

    # Step 1: Enhance contrast using CLAHE
    img1_enhanced = enhance_contrast_clahe(img1)
    img2_enhanced = enhance_contrast_clahe(img2)
    print("Contrast enhanced for both images.")

    # Step 2: Detect features using SIFT with adjusted parameters
    keypoints1, descriptors1 = detect_features_sift(img1_enhanced, nfeatures, contrastThreshold, edgeThreshold, sigma)
    keypoints2, descriptors2 = detect_features_sift(img2_enhanced, nfeatures, contrastThreshold, edgeThreshold, sigma)
    print("Feature detection completed for both images.")

    # Step 3: Match features between the two images
    matches = match_features(descriptors1, descriptors2)
    print(f"Found {len(matches)} matches.")

    # Step 4: Calculate mean change in pixel distance
    mean_change = calculate_mean_change(matches, keypoints1, keypoints2)
    print(f"Mean change in pixel distance: {mean_change}")

    # Optional: Draw matches for visualization
    img_matches = cv2.drawMatches(img1_enhanced, keypoints1, img2_enhanced, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display the matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done.")
