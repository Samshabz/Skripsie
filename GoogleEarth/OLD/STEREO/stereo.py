import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import os  # For file operations
import matplotlib.pyplot as plt  # For plotting
import torch  # For deep learning
from lightglue import LightGlue, SuperPoint  # For feature extraction
from lightglue.utils import rbd  # Utility function for removing batch dimension

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if a GPU is available

# Initialize SuperPoint and LightGlue
extractor = SuperPoint(
    max_num_keypoints=2048,  # Maximum number of keypoints
    detection_threshold=0.0000015,  # Detection threshold
    nms_radius=5  # Non-maximum suppression radius
).eval().to(device)  # Set the model to evaluation mode and move it to the device

matcher = LightGlue(  # Initialize LightGlue
    features='superpoint',  # Use SuperPoint features
    depth_confidence=0.95,
    width_confidence=0.99,
    filter_threshold=0.015  # Custom filter threshold
).eval().to(device)

def normalize_image(image):
    """Normalize the image to [0,1] and convert to torch tensor."""
    return torch.tensor(image / 255.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)

def detect_and_compute_superpoint(image):
    """Detect and compute SuperPoint features."""
    normalized_image = normalize_image(image)
    feats = extractor.extract(normalized_image)
    return feats

def match_features_superpoint(featsA, featsB):
    """Match features using LightGlue."""
    matches = matcher({'image0': featsA, 'image1': featsB})
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]
    return featsA, featsB, matches['matches']

class ImageAlignerAndDepthMap:
    def __init__(self):
        pass

    def crop_image(self, image):
        """Crop the top and bottom 10% and left and right 2% of the image."""
        height, width = image.shape[:2]
        cropped_image = image[int(height * 0.1):int(height * 0.9), int(width * 0.02):int(width * 0.98)]
        return cropped_image

    def align_images(self, img_left, img_right):
        # Convert images to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Detect and compute keypoints and descriptors using SuperPoint
        feats_left = detect_and_compute_superpoint(gray_left)
        feats_right = detect_and_compute_superpoint(gray_right)

        if feats_left is None or feats_right is None:
            print("Error: No descriptors found. Ensure the images have enough features.")
            return None, None, None, None

        # Match features using LightGlue
        feats_left, feats_right, matches = match_features_superpoint(feats_left, feats_right)
        if len(matches) < 4:
            print("Error: Not enough good matches found.")
            return None, None, None, None

        # Extract matched keypoints
        keypoints_left = feats_left['keypoints'].cpu().numpy()
        keypoints_right = feats_right['keypoints'].cpu().numpy()
        src_pts = keypoints_left[matches[:, 0]]
        dst_pts = keypoints_right[matches[:, 1]]
        shifts = dst_pts - src_pts

        # Calculate the mean translation (shift) for x and y
        mean_shift_x = -np.mean(shifts[:, 0])
        mean_shift_y = -np.mean(shifts[:, 1])

        # Calculate the rotation matrix and angle
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))

        print(f"Rotation Angle: {angle:.2f} degrees")
        print(f"Mean Shift: ({mean_shift_x:.2f}, {mean_shift_y:.2f}) pixels")

        # Apply the rotation to the right image
        rows, cols = img_right.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_img_right = cv2.warpAffine(img_right, rotation_matrix, (cols, rows))

        # Apply the translation manually
        translated_img_right = np.zeros_like(rotated_img_right)
        translation_x = int(mean_shift_x)
        translation_y = int(mean_shift_y)

        # Handle different cases for positive and negative shifts
        if translation_x >= 0 and translation_y >= 0:
            translated_img_right[translation_y:, translation_x:] = rotated_img_right[:-translation_y or None, :-translation_x or None]
        elif translation_x >= 0 and translation_y < 0:
            translated_img_right[:translation_y, translation_x:] = rotated_img_right[-translation_y:, :-translation_x or None]
        elif translation_x < 0 and translation_y >= 0:
            translated_img_right[translation_y:, :translation_x] = rotated_img_right[:-translation_y or None, -translation_x:]
        else:
            translated_img_right[:translation_y, :translation_x] = rotated_img_right[-translation_y:, -translation_x:]

        return img_left, translated_img_right, keypoints_left, keypoints_right, matches

    def overlay_matches(self, img_left, img_right, keypoints_left, keypoints_right, good_matches):
        # Draw matches on the images
        match_img = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the image with matches
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title("Keypoint Matches After Alignment")
        plt.axis('off')
        plt.show()

    def compute_pixel_shift_heatmap(self, img_left, img_right, keypoints_left, keypoints_right, good_matches):
        # Create a blank image for the heatmap
        heatmap = np.zeros_like(img_left[:, :, 0], dtype=np.float32)

        for match in good_matches:
            # Get coordinates of matched keypoints
            pt1 = np.int32(keypoints_left[match[0]])
            pt2 = np.int32(keypoints_right[match[1]])

            # Calculate the shift between the matched points
            shift = np.linalg.norm(pt1 - pt2)

            # Apply logarithmic scaling to the shift
            shift_log = np.log1p(shift)

            # Plot a 5x5 square on the heatmap at the location of the keypoint from the left image
            heatmap[pt1[1] - 4:pt1[1] + 5, pt1[0] - 4:pt1[0] + 5] = shift_log

        # Clip the heatmap to avoid extreme values
        heatmap = np.clip(heatmap, np.percentile(heatmap, 5), np.percentile(heatmap, 95))

        # Normalize the heatmap for visualization
        heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)

        # Display the heatmap
        plt.figure(figsize=(15, 10))
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar()
        plt.title("Pixel Shift Heatmap")
        plt.axis('off')
        plt.show()

    def overlay_aligned(self, img_left, img_right):
        # Convert to colored images for visualization
        img_left_colored = img_left
        img_right_colored = img_right

        img_left_colored[:, :, 1:3] = 0  # Remove green and blue channels (red overlay)
        img_right_colored[:, :, 0] = 0  # Remove red channel (cyan overlay)

        overlay = cv2.addWeighted(img_left_colored, 0.5, img_right_colored, 0.5, 0)

        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlayed Aligned Images")
        plt.axis('off')
        plt.show()

    def compute_depth_map(self, img_left, img_right):
        try_disp = [7, 10, 14]  # Depth ranges
        try_window_size = [7]  # Smoothness settings

        for disp in try_disp:
            for window_size in try_window_size:
                num_disp = 16 * disp  # Number of disparities

                # Stereo SGBM setup
                stereo = cv2.StereoSGBM_create(minDisparity=0,
                                            numDisparities=num_disp,
                                            blockSize=window_size,
                                            P1=8 * 3 * window_size ** 2,
                                            P2=32 * 3 * window_size ** 2,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=15,
                                            speckleWindowSize=0,
                                            speckleRange=2,
                                            preFilterCap=63,
                                            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

                disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

                # Clip the disparity values to reduce extremes
                disparity = np.clip(disparity, 0, num_disp)

                # Normalize the disparity map within the clipped range for visualization
                disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                disparity_normalized = np.uint8(disparity_normalized)

                # Apply histogram equalization to improve contrast
                disparity_equalized = cv2.equalizeHist(disparity_normalized)

                # Display the disparity map
                plt.imshow(disparity_equalized, cmap='plasma')
                plt.colorbar()
                plt.title(f'Disp (SGBM), Window = {window_size}, Disp = {disp}')
                plt.show()

        return disparity


def main():
    aligner = ImageAlignerAndDepthMap()
    root = './GOOGLEearth/stereo/'  # Replace with your path
    imgLeftPath = os.path.join(root, 'close3.jpg')
    imgRightPath = os.path.join(root, 'close4.jpg')
    
    # Read images
    img_left = cv2.imread(imgLeftPath)
    img_right = cv2.imread(imgRightPath)
    
    # Crop the images
    img_left = aligner.crop_image(img_left)
    img_right = aligner.crop_image(img_right)

    # Align images using rotation and mean pixel shifts
    img_left_aligned, img_right_aligned, keypoints_left, keypoints_right, good_matches = aligner.align_images(img_left, img_right)

    if img_left_aligned is not None and img_right_aligned is not None:
        # Overlay the aligned images for visual verification
        aligner.overlay_aligned(img_left_aligned, img_right_aligned)

        # Generate and display the pixel shift heatmap
        aligner.compute_pixel_shift_heatmap(img_left_aligned, img_right_aligned, keypoints_left, keypoints_right, good_matches)

        # Compute the depth map
        aligner.compute_depth_map(img_left_aligned, img_right_aligned)

if __name__ == "__main__":
    main()
