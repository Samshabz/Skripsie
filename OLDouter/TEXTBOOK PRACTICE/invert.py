import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
image_path = 'Square.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Compute the 2D Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)  # Shift zero frequency component to the center
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Plot the input image
plt.figure(figsize=(12, 6))

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

# Plot the magnitude spectrum
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xlabel('Horizontal Frequency')
plt.ylabel('Vertical Frequency')
plt.colorbar(label='Magnitude')

plt.show()
