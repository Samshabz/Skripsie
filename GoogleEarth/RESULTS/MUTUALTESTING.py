import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the table for plotting
mutual_overlap = [
    75.018, 70.635, 66.404, 62.283, 58.314, 55.381, 51.568, 47.945, 44.471, 
    41.110, 37.898, 34.800, 31.296, 28.562, 25.819, 23.433, 18.689, 18.153, 15.940
]
radial_percent = [
    0.2224, 0.2225, 0.2227, 0.2235, 0.2238, 0.2227, 0.2218, 0.2188, 0.2168, 
    0.2176, 0.2270, 0.2377, 0.2305, 0.2583, 0.2630, 0.5935, 14.9185, 0.3305, 0.3355
]
location_inference_time = [
    1.3877, 1.5190, 1.7548, 1.3438, 1.3458, 1.4385, 2.0892, 1.1790, 1.1468, 
    1.0791, 1.2183, 1.0016, 0.9484, 0.8207, 0.7982, 0.7596, 0.7338, 0.9540, 1.0185
]

# Setting font size globally for the plot
plt.rcParams.update({'font.size': 12})

# Testing a different threshold value in the symmetrical log scale to stretch the main values more effectively.
# Adjusting linthresh to 0.5 to allow a broader separation of smaller values while still managing the outlier.

plt.figure(figsize=(10, 6))
plt.plot(mutual_overlap, radial_percent, marker='o', linestyle='-', color='b')
plt.yscale('log')  # Increasing linthresh to 0.5 for greater impact on spreading smaller values
plt.xlabel('Total Mutual Overlap (%)', fontsize=12)
plt.ylabel('Mean Radial Percent (%)', fontsize=12)
plt.title('Total Mutual Overlap vs Mean Radial Percent Error', fontsize=14)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.show()

import os

directory = "./REPORT/stellenbosch_ee_report_template-master/Chapter 5/RESULTPLOTS"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

file_name = f"{directory}/MUTUAL_ACC.png"
plt.savefig(file_name)  # Save the figure as a PNG file
plt.close()  # Close the figure

# Plotting Mean Location Inference Time
plt.figure(figsize=(10, 6))
plt.plot(mutual_overlap, location_inference_time, marker='s', linestyle='-', color='r')
plt.xlabel('Total Mutual Overlap (%)')
plt.ylabel('Mean Location Inference Time (s)')
plt.title('Total Mutual Overlap vs Mean Location Inference Time')
plt.grid(True, linestyle='--', linewidth=0.5)


directory = "./REPORT/stellenbosch_ee_report_template-master/Chapter 5/RESULTPLOTS"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

file_name = f"{directory}/MUTUAL_TIME.png"
plt.savefig(file_name)  # Save the figure as a PNG file
plt.close()  # Close the figure
