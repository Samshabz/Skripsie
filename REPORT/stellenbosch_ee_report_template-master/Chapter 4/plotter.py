import matplotlib.pyplot as plt
import numpy as np

# Define the keypoint targets
keypoints = [800, 1100, 1500, 2500, 4000, 6000]

# MAE GPS Errors for BFMatcher across datasets
bf_errors = {
    "CITY1": [390.54, 202.65, 221.28, 61.48, 59.07, 59.18],
    "CITY2": [398.80, 120.48, 270.63, 14.33, 6.18, 5.12],
    "ROCKY": [373.43, 240.33, 315.50, 112.25, 37.57, 16.74],
    "DESERT": [388.44, 170.46, 271.03, 98.61, 88.54, 77.05],
    "AMAZON": [293.49, 124.54, 166.31, 46.73, 37.68, 35.37]
}

# MAE GPS Errors for FLANN across datasets
flann_errors = {
    "CITY1": [460.49, 259.00, 199.07, 67.94, 57.72, 52.29],
    "CITY2": [422.72, 142.90, 294.84, 19.08, 5.44, 4.81],
    "ROCKY": [400.24, 237.92, 325.96, 124.91, 52.76, 19.26],
    "DESERT": [422.86, 173.37, 305.29, 107.08, 91.40, 76.70],
    "AMAZON": [354.26, 127.16, 184.34, 47.27, 38.49, 34.67]
}

# Calculate the divergence (BFMatcher - FLANN)
divergences = {
    dataset: np.array(flann_errors[dataset]) - np.array(bf_errors[dataset])
    for dataset in bf_errors
}

# Plotting the divergences
plt.figure(figsize=(12, 8))

for dataset, divergence in divergences.items():
    plt.plot(keypoints, divergence, label=f"{dataset}", marker='o')

# Labels and Title
plt.xlabel('Keypoints Target', fontsize=38)
plt.ylabel('Radial RMSE Difference (m)', fontsize=38)
plt.title('Matcher Difference (FLANN-BF) At Varying Keypoint Targets', fontsize=38)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.legend(loc='upper right', fontsize=26)
plt.grid(True)

# Show Plot
plt.tight_layout()
plt.show()
