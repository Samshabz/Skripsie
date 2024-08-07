import cv2
import numpy as np
import matplotlib.pyplot as plt

def automatic_image_correction(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Parameters (these need to be determined by calibration)
    # Example calibration parameters; these would normally be determined through a calibration process
    camera_matrix = np.array([[1.2e3, 0, 6.4e2], [0, 1.2e3, 3.6e2], [0, 0, 1]])
    dist_coeffs = np.array([-0.3, 0.1, 0, 0])

    # Undistort the image (radial and tangential distortion)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    print(f"Undistortion complete for {image_path}.")

    # Vignetting correction
    rows, cols = undistorted_image.shape[:2]
    X, Y = np.ogrid[:rows, :cols]
    center_x, center_y = rows / 2, cols / 2
    radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_radius = np.sqrt(center_x**2 + center_y**2)
    vignette_strength = 0.5
    gain_map = 1 + (1 - radius / max_radius) * vignette_strength
    gain_map = gain_map[:, :, np.newaxis]  # Add a new axis to match the image shape

    # Apply the gain map to the image
    vignetting_corrected_image = undistorted_image * gain_map
    print(f"Vignetting correction complete for {image_path}.")

    # Normalize the corrected image to be in the range [0, 255]
    vignetting_corrected_image = np.clip(vignetting_corrected_image, 0, 255).astype(np.uint8)

    # Save the corrected image
    corrected_image_bgr = cv2.cvtColor(vignetting_corrected_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, corrected_image_bgr)
    print(f"Saved corrected image to {output_path}.")

    return image, vignetting_corrected_image

# Paths to the original images and the output images
image_paths = ['./RESOURCES/imageC0.png', './RESOURCES/imageC1.png']
output_paths = ['./RESOURCES/CC0.png', './RESOURCES/CC1.png']

# Process and save the images
for image_path, output_path in zip(image_paths, output_paths):
    original_image, corrected_image = automatic_image_correction(image_path, output_path)

    # Display the original and corrected images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title('Corrected Image')
    plt.imshow(corrected_image)
    plt.show()
