# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
# day = np.array([0.362, 0.219, 1.909, 0.251, 0.381])
# early_eve = np.array([0.363, 0.222, 2.071, 0.425, 0.578])
# late_eve = np.array([0.365, 0.225, 2.290, 28.713, 141.647])
# midnight = np.array([0.371, 0.310, 20.399, 112.610, 153.678])

# # Calculate percentage increase from baseline (Day)
# early_eve_increase = ((early_eve - day) / day) * 100
# late_eve_increase = ((late_eve - day) / day) * 100
# midnight_increase = ((midnight - day) / day) * 100

# # Bar width and positioning
# bar_width = 0.3
# index = np.arange(len(datasets))

# # Plotting
# fig, ax = plt.subplots(figsize=(12, 7))
# ax.bar(index, early_eve_increase, bar_width, label='Early Eve', color='lightgreen')
# ax.bar(index + bar_width, late_eve_increase, bar_width, label='Late Eve', color='gold')
# ax.bar(index + 2 * bar_width, midnight_increase, bar_width, label='Midnight', color='lightcoral')

# # Labels and title
# ax.set_xlabel('Dataset', fontsize=22)
# ax.set_ylabel('Error Increase From Daytime (%, Log)', fontsize=22)
# ax.set_title('Increase in Error for Various Times', fontsize=26)
# ax.set_xticks(index + bar_width)
# ax.set_xticklabels(datasets, fontsize=16)

# # Legend and layout adjustments
# ax.legend(fontsize=22)
# plt.yticks(fontsize=16)
# plt.yscale('symlog')
# plt.xticks(fontsize=16)
# plt.tight_layout()


# import os
# directory = "./REPORT/stellenbosch_ee_report_template-master/Chapter 5/RESULTPLOTS/lighting"

# os.makedirs(directory, exist_ok=True)

# file_name = f"{directory}/lightresults.png"
# plt.savefig(file_name)  # Save the figure as a PNG file
# plt.close()  # Close the figure

# Show the plot
# plt.show()



# below is not stacked. its  low res test

import matplotlib.pyplot as plt
import numpy as np

# Data
resolutions = [
    (1920 * 972), (1536 * 777.6), (768 * 388.8), (576 * 291.6), (528 * 347.49),
    (480 * 315.9), (432 * 284.31), (384 * 252.72), (336 * 221.13)
]
base_resolution = 1920 * 972

# Calculating overlap as a fraction of the base resolution
overlap = np.array([res / base_resolution for res in resolutions])

# Mean radial percentages for each dataset at each resolution
data_errors = {
    'CITY1': [0.4218, 0.4283, 0.4506, 0.5819, 6.8508, 8.0347, 11.9210, 22.2008, 29.2928],
    'CITY2': [0.2216, 0.2298, 0.2393, 0.2505, 12.1395, 12.1405, 12.2489, 14.5228, 20.0428],
    'ROCKY': [80.2173, 1.4441, 1.6923, 1.8616, 2.0240, 51.6151, 46.3796, 51.2201, 73.3333],
    'DESERT': [20.8159, 0.2945, 0.3325, 0.2974, 13.2917, 12.2575, 24.0492, 14.5317, 26.0281],
    'AMAZON': [30.3264, 18.2228, 0.6585, 18.0415, 24.7928, 35.0960, 69.1683, 82.1270, 354.0224]
}

data_runtimes = {
    'CITY1': [1.5190, 1.6812, 0.8300, 0.4359, 0.4015, 0.4110, 0.3779, 0.3591, 0.3452],
    'CITY2': [1.7689, 1.6480, 0.6818, 0.4863, 0.4111, 0.3987, 0.3669, 0.3273, 0.3367],
    'ROCKY': [2.0976, 1.8400, 0.7002, 0.7214, 0.4917, 0.3572, 0.3670, 0.3496, 0.3020],
    'DESERT': [2.1469, 1.5383, 0.6885, 0.3736, 0.3159, 0.2913, 0.2324, 0.2088, 0.2179],
    'AMAZON': [2.1608, 1.8499, 0.6561, 0.3592, 0.3876, 0.3423, 0.3214, 0.3068, 0.3091]
}

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))
for dataset, errors in data_errors.items():
    ax.plot(overlap, errors, label=dataset, linewidth=2, marker='o')  # s sets the marker size

# Labels and title
ax.set_xlabel('Resolution Factor (1920x972 Baseline)', fontsize=26)
ax.set_ylabel('Mean Radial Percent Error (%)', fontsize=26)
ax.set_title('Mean Radial Percent Error vs Resolution Factor', fontsize=28)
ax.legend(fontsize=24)
plt.grid(True)
plt.xticks(fontsize=20)
plt.xscale('log')
plt.yticks(fontsize=20)
plt.tight_layout()

import os
directory = "./REPORT/stellenbosch_ee_report_template-master/Chapter 5/RESULTPLOTS/lowres"
os.makedirs(directory, exist_ok=True)
file_name = f"{directory}/lowresacc.png"
plt.savefig(file_name)  # Save the figure as a PNG file
plt.close()  # Close the figure


# Show the plot
# plt.show()


# Plot for runtime
fig, ax = plt.subplots(figsize=(14, 8))
for dataset, runtimes in data_runtimes.items():
    ax.plot(overlap, runtimes, label=dataset, linewidth=2, marker='o')

# Labels and title
ax.set_xlabel('Resolution Factor (1920x972 Baseline)', fontsize=26)
ax.set_ylabel('Mean Runtime (s)', fontsize=26)
ax.set_title('Mean Runtime vs Overlap for Resolution Factor', fontsize=28)
ax.legend(fontsize=24)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

# import os
directory = "./REPORT/stellenbosch_ee_report_template-master/Chapter 5/RESULTPLOTS/lowres"
os.makedirs(directory, exist_ok=True)
file_name = f"{directory}/lowrestime.png"
plt.savefig(file_name)  # Save the figure as a PNG file
plt.close()  # Close the figure


# Show the plot
# plt.show()





# KEY METRICS START

# import matplotlib.pyplot as plt

# # Data for radial errors
# datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
# mean_radial_errors = [0.4001, 0.2290, 1.9154, 0.2515, 0.3811]
# max_radial_errors = [0.6356, 0.2715, 6.2123, 0.3516, 0.7447]

# # Data for processing times
# mean_add_time = [0.5603, 0.5454, 0.4912, 0.4171, 0.5314]
# max_add_time = [0.7671, 0.6438, 0.6098, 0.5649, 0.6731]
# mean_param_time = [1.3363, 1.2972, 1.2658, 1.3197, 1.2271]
# max_param_time = [1.2706, 1.4904, 1.6314, 1.6769, 1.7244]
# mean_loc_time = [1.2028, 1.2866, 1.1829, 1.2862, 1.2837]
# max_loc_time = [1.3609, 1.6271, 1.4374, 1.8392, 1.7091]

# # Plot 1: Radial Errors
# plt.figure(figsize=(14, 6))
# plt.plot(datasets, mean_radial_errors, marker='o', label='Mean Radial Error (%)')
# plt.plot(datasets, max_radial_errors, marker='o', label='Max Radial Error (%)')
# plt.title('Mean and Maximum Radial Errors', fontsize=22)
# plt.xlabel('Dataset', fontsize=14)
# plt.ylabel('Percentage Radial Error (%)', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(fontsize=16)
# plt.grid(True)
# import os

# directory = "./REPORT/stellenbosch_ee_report_template-master/Chapter 5/RESULTPLOTS/newkey"

# # Create the directory if it doesn't exist
# os.makedirs(directory, exist_ok=True)

# file_name = f"{directory}/percacc.png"
# plt.savefig(file_name)  # Save the figure as a PNG file
# plt.close()  # Close the figure



# # plt.show()

# # Plot 2: Processing Times
# plt.figure(figsize=(14, 6))
# plt.plot(datasets, mean_add_time, marker='o', label='Mean Add Time')
# plt.plot(datasets, max_add_time, marker='o', label='Max Add Time')
# plt.plot(datasets, mean_param_time, marker='o', label='Mean Parameter Inference Time')
# plt.plot(datasets, max_param_time, marker='o', label='Max Parameter Inference Time')
# plt.plot(datasets, mean_loc_time, marker='o', label='Mean Location Inference Time')
# plt.plot(datasets, max_loc_time, marker='o', label='Max Location Inference Time')
# plt.ylim(0.35,2.5)
# plt.title('Mean and Maximum Processing Times (Seconds)', fontsize=22)
# plt.xlabel('Dataset', fontsize=20)
# plt.ylabel('Time (Seconds)', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(loc='upper left', fontsize=13)
# plt.grid(True)
# # plt.show()

# file_name = f"{directory}/pixmetacc.png"
# plt.savefig(file_name)  # Save the figure as a PNG file
# plt.close()  # Close the figure




# # Data for pixel and meter errors
# mean_pixel_error = [0.8196, 0.6114, 1.0520, 0.5668, 0.3919]
# max_pixel_error = [2.2948, 0.9949, 4.9280, 1.7829, 1.0508]
# mean_meter_error = [4.7978, 2.7576, 5.8822, 2.3573, 2.5336]
# max_meter_error = [13.4077, 4.4886, 27.5392, 7.4109, 6.7927]

# fig, ax1 = plt.subplots(figsize=(14, 6))

# # Plotting Pixel Errors on primary y-axis
# ax1.plot(datasets, mean_pixel_error, marker='o', color='blue', label='Mean Pixel Error')
# ax1.plot(datasets, max_pixel_error, marker='o', color='lightblue', label='Max Pixel Error')
# ax1.set_xlabel('Dataset', fontsize=20)
# ax1.set_ylabel('Pixel Error', color='blue', fontsize=20)
# ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
# ax1.tick_params(axis='x', labelsize=16)
# ax1.grid(True)

# # Creating a second y-axis for Meter Errors
# ax2 = ax1.twinx()
# ax2.plot(datasets, mean_meter_error, marker='o', color='green', label='Mean Meter Error')
# ax2.plot(datasets, max_meter_error, marker='o', color='lightgreen', label='Max Meter Error')
# ax2.set_ylabel('Meter Error', color='green', fontsize=20)
# ax2.tick_params(axis='y', labelcolor='green', labelsize=16)

# # Adding title and legend
# plt.title('Mean and Maximum Pixel and Meter Errors', fontsize=22)
# fig.tight_layout()
# fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=16)

# # plt.show()
# file_name = f"{directory}/timeall.png"
# plt.savefig(file_name)  # Save the figure as a PNG file
# plt.close()  # Close the figure

