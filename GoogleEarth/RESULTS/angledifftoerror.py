import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# For Dataset DATSETROT
angle_diff_DATSETROT = [
    0.1320, 0.1079, 0.1015, 0.0177, 0.0385, 3.9354, 4.2739, 
    0.0515, 0.0159, 0.1577, 0.0421, 0.0927, 0.0464
]

dev_x_DATSETROT = [
    2.9678, 4.2594, 22.2764, 19.1151, 15.5839, 13.2910, 
    8.7520, 1.3706, 2.1571, 4.9120, 7.4819, 0.1969, 8.2002
]

dev_y_DATSETROT = [
    17.0333, 16.9070, 0.5338, 17.1488, 17.9078, 13.3794, 
    7.8536, 0.3114, 0.3491, 12.1787, 14.0519, 11.9434, 6.7959
]

# For Dataset DATSETCPT
angle_diff_DATSETCPT = [
    0.4335, 0.2436, 0.1056, 0.1153, 0.1015, 0.2880, 
    0.0838, 0.3321, 0.6878, 0.4014, 0.4051, 0.9985, 0.0766, 0.2450
]

dev_x_DATSETCPT = [
    10.1515, 5.0534, 0.0052, 0.8845, 0.6374, 1.4214, 
    2.0925, 0.1675, 0.9139, 0.6227, 1.3504, 3.3827, 3.0927, 0.4858
]

dev_y_DATSETCPT = [
    1.6780, 2.8720, 2.5643, 2.1262, 0.7839, 2.3174, 
    1.4449, 0.6843, 0.9162, 0.4796, 2.6679, 1.0026, 0.3766, 2.7051
]

# For Dataset DATSETROCK
angle_diff_DATSETROCK = [
    0.0663, 0.6339, 0.0851, 0.9888, 0.3564, 0.9093, 
    0.2771, 0.0426, 0.0452, 1.9337, 0.4338, 0.0136, 0.0105
]

dev_x_DATSETROCK = [
    0.4147, 6.4142, 0.4974, 8.9464, 7.1414, 3.4425, 
    5.5636, 0.2767, 0.2747, 7.0683, 4.9601, 6.6895, 6.6065
]

dev_y_DATSETROCK = [
    0.5569, 0.4169, 0.5960, 3.1007, 14.8995, 8.1955, 
    13.0448, 0.7099, 0.6902, 8.7854, 6.9348, 3.2521, 3.1686
]

# For Dataset DATSETSAND
angle_diff_DATSETSAND = [
    0.3123, 0.2870, 0.7002, 1.7740, 0.8558, 0.3289, 
    0.6892, 0.8956, 0.1337, 0.1882, 0.1786, 0.1082, 0.8371
]

dev_x_DATSETSAND = [
    14.3055, 12.7643, 6.7444, 4.6957, 12.8777, 8.7477, 
    7.4343, 7.4178, 6.8971, 6.3894, 7.8691, 5.7628, 20.2830
]

dev_y_DATSETSAND = [
    0.9728, 0.5533, 0.6218, 4.6911, 7.6073, 1.2640, 
    7.7949, 6.3976, 5.1170, 4.5870, 5.6084, 8.3531, 7.1605
]

# For Dataset DATSETAMAZ
angle_diff_DATSETAMAZ = [
    357.0421, 2.0155, 0.3960, 0.0183, 1.3750, 0.8859, 
    0.7678, 2.1278, 2.2649, 2.3206
]

dev_x_DATSETAMAZ = [
    4.4377, 3.8383, 8.2464, 12.0707, 2.9783, 6.9897, 
    7.9108, 4.8379, 0.9918, 1.4751
]

dev_y_DATSETAMAZ = [
    2.3340, 19.9010, 3.8290, 5.7311, 5.7334, 2.9872, 
    3.1328, 5.8005, 16.2729, 3.4825
]


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Function to calculate radial pixel error
def calculate_radial_differences(x_dev, y_dev):
    return np.sqrt(np.array(x_dev)**2 + np.array(y_dev)**2)

# Dataset: DATSETROT
radial_diff_DATSETROT = calculate_radial_differences(dev_x_DATSETROT, dev_y_DATSETROT)
# Dataset: DATSETCPT
radial_diff_DATSETCPT = calculate_radial_differences(dev_x_DATSETCPT, dev_y_DATSETCPT)
# Dataset: DATSETROCK
radial_diff_DATSETROCK = calculate_radial_differences(dev_x_DATSETROCK, dev_y_DATSETROCK)
# Dataset: DATSETSAND
radial_diff_DATSETSAND = calculate_radial_differences(dev_x_DATSETSAND, dev_y_DATSETSAND)
# Dataset: DATSETAMAZ
radial_diff_DATSETAMAZ = calculate_radial_differences(dev_x_DATSETAMAZ, dev_y_DATSETAMAZ)

# Combine all angles and radial differences
angles_all = angle_diff_DATSETROT + angle_diff_DATSETCPT + angle_diff_DATSETROCK + angle_diff_DATSETSAND + angle_diff_DATSETAMAZ
radial_diff_all = np.concatenate([radial_diff_DATSETROT, radial_diff_DATSETCPT, radial_diff_DATSETROCK, radial_diff_DATSETSAND, radial_diff_DATSETAMAZ])

# Calculate the Pearson correlation coefficient
correlation, _ = pearsonr(angles_all, radial_diff_all)

# Create the scatter plot
plt.figure(figsize=(10, 6))
# limit x and y to -5 to 5
plt.xlim(-1, 6)
plt.scatter(angles_all, radial_diff_all, label='Radial Pixel Errors', alpha=0.7)
plt.title('Scatter Plot of Internal Angle vs Radial Pixel Error')
plt.xlabel('Internal Angle Difference (degrees)')
plt.ylabel('Radial Pixel Error')
plt.grid(True)

# Display the correlation on the plot
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', ha='right', va='center', transform=plt.gca().transAxes, fontsize=12, color='red')

# Show the plot
plt.show()