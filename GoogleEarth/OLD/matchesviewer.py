import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class ImageMatcher:
    def __init__(self):
        # Initialize the SIFT detector
        self.sift = cv2.SIFT_create()

    def crop_image(self, image):
        """Crop the top and bottom 10% of the image."""
        height = image.shape[0]
        cropped_image = image[int(height * 0.075):int(height * 0.925), :]
        return cropped_image

    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors."""
        cropped_image = self.crop_image(image)
        keypoints, descriptors = self.sift.detectAndCompute(cropped_image, None)
        return cropped_image, keypoints, descriptors

    def match_images(self, image1, image2):
        """Match features between two images, draw matches, and plot pixel changes."""
        cropped_image1, keypoints1, descriptors1 = self.detect_and_compute(image1)
        cropped_image2, keypoints2, descriptors2 = self.detect_and_compute(image2)

        # BFMatcher with default params and cross-check
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by the distance - best matches first
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw top 50 matches
        match_image = cv2.drawMatches(
            cropped_image1, keypoints1, cropped_image2, keypoints2, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Convert BGR image to RGB for matplotlib
        match_image_rgb = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)

        # Display the matches
        plt.figure(figsize=(15, 10))
        plt.imshow(match_image_rgb)
        plt.title('Top 50 Feature Matches between Image 1 and Image 2')
        plt.axis('off')
        plt.show()

        # Calculate pixel changes
        pixel_changes = [np.linalg.norm(np.array(keypoints1[m.queryIdx].pt) - np.array(keypoints2[m.trainIdx].pt)) for m in matches]
        rounded_pixel_changes = np.round(pixel_changes)  # Round to nearest pixel

        # Remove outliers or limit range for histogram
        percentile_5 = np.percentile(rounded_pixel_changes, 20)
        percentile_95 = np.percentile(rounded_pixel_changes, 80)

        # Plot the distribution curve within a limited range
        plt.figure(figsize=(10, 5))
        plt.hist(rounded_pixel_changes, bins=np.arange(percentile_5, percentile_95 + 1, 1), alpha=0.75, color='blue')
        plt.title('Distribution of Pixel Changes (5th to 95th Percentiles)')
        plt.xlabel('Pixel Change (rounded)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def main():
    directory = './GoogleEarth/SET1'
    image_path1 = os.path.join(directory, '1.jpg')
    image_path2 = os.path.join(directory, '2.jpg')

    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    matcher = ImageMatcher()
    matcher.match_images(image1, image2)

if __name__ == "__main__":
    main()
