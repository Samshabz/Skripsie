import numpy as np
import matplotlib.pyplot as plt

# Provided absolute clean data points
x_clean = np.array([1301, 1301, 1301, 1091, 1140, 1091, 1091, 1140, 1091, 1140, 879, 925, 
                    879, 879, 1301, 987, 879, 717, 1091, 669, 669, 879, 669, 668, 879, 669, 
                    457, 457, 457, 338, 307, 331, 331, 248, 331, 238, 243, 346, 420, 328, 
                    212, 310, 164, 208])
y_clean = np.array([526, 426, 324, 524, 244, 425, 324, 179, 224, 147, 524, 247, 424, 324, 
                    236, 147, 224, 155, 236, 523, 424, 145, 324, 218, 235, 172, 244, 263, 
                    235, 525, 416, 223, 316, 317, 233, 525, 424, 494, 436, 408, 344, 263, 
                    346, 263])
percentage_deviation_clean = np.array([0.78, 0.78, 0.92, 0.79, 0.84, 0.69, 0.67, 15.40, 
                                       2.22, 31.71, 0.78, 44.97, 0.75, 0.77, 3.93, 3.08, 
                                       1.21, 4.50, 98.99, 0.86, 0.81, 53.75, 0.69, 15.07, 
                                       96.60, 22.80, 41.72, 81.63, 89.87, 67.05, 48.65, 32.45, 
                                       75.45, 86.40, 96.49, 71.21, 143.72, 85.39, 69.53, 
                                       157.80, 112.33, 176.16, 108.86, 161.50])

# Define the provided time values in a structured manner (as tuples)
time_data = {
    (1301, 526): 37.8268,
    (1301, 426): 36.5297,
    (1301, 324): 37.2607,
    (1091, 524): 40.6775,
    (1140, 244): 40.1711,
    (1091, 425): 39.0059,
    (1091, 324): 48.7385,
    (1140, 179): 29.9646,
    (1140, 224): 36.5297,
    (1140, 147): 29.8089,
    (879, 524): 42.5109,
    (879, 424): 36.7949,
    (1091, 524): 37.8268,
    (1091, 324): 21.1185,  
    (879, 324): 42.5109,
    (879, 224): 21.1185,
    (1140, 147): 30.9306,
    (879, 524): 21.0493,
    (879, 424): 23.7240,
    (879, 224): 21.6485,
    (1091, 324): 18.5804,
    (1091, 224): 16.0939,
    (1091, 147): 16.2237,
    (1301, 236): 32.0265,
    (987, 147): 34.7769,
    (1091, 224): 38.9822,
    (717, 155): 31.4418,
    (1091, 236): 23.2089,
    (669, 523): 27.6688,
    (669, 424): 42.3228,
    (879, 145): 24.5935,
    (669, 324): 40.3082,
    (668, 218): 23.2824,
    (879, 235): 20.3500,
    (669, 172): 26.4315,
    (457, 524): 32.9186,
    (457, 423): 28.5447,
    (669, 235): 21.8691,
    (457, 328): 31.8120,
    (457, 225): 25.5036,
    (457, 244): 16.6794,
    (457, 263): 15.2605,
    (457, 235): 13.3438,
    (338, 525): 20.9649,
    (307, 416): 19.5704,
    (248, 325): 20.4010,
    (331, 223): 20.2815,
    (331, 316): 18.1319,
    (248, 317): 15.5262,
    (331, 233): 14.6083,
    (238, 525): 17.2091,
    (243, 424): 16.4516,
    (346, 494): 14.8200,
    (420, 436): 14.7691,
    (328, 408): 13.1715,
    (212, 344): 14.4940,
    (310, 263): 11.4212,
    (168, 523): 14.8554,
    (225, 720): 13.7296,
    (224, 560): 12.3333,
    (219, 461): 12.5729,
    (182, 418): 11.3477,
    (164, 346): 10.7306,
}

# Lookup time values corresponding to the provided clean xy points
time_values = np.array([time_data.get((x, y), None) for x, y in zip(x_clean, y_clean)])

# Calculate Mutual Information
mutual_information = x_clean * y_clean  # Example calculation for mutual information

# Calculate Accuracy
accuracy = percentage_deviation_clean

# First Scatter Plot with Time as Color for Mutual Information vs Accuracy
plt.figure(figsize=(12, 10))
scatter_mi_acc = plt.scatter(
    accuracy,
    mutual_information,
    c=time_values,        # Set time as color
    cmap='plasma_r',      # Colormap for visualization
    edgecolor='k',
    alpha=0.7,
    s=100  # Adjust as needed
)

plt.colorbar(label='Time (s)')  # Add colorbar for time
plt.xlabel('Deviation (%)', fontsize=14)
plt.ylabel('Mutual Information', fontsize=14)
plt.title('Scatter Plot of Deviation vs Mutual Information', fontsize=16)
plt.grid(True)
plt.tight_layout()

# Second Scatter Plot for Accuracy vs Mean Total Time
plt.figure(figsize=(12, 10))
scatter_acc_time = plt.scatter(
    accuracy,
    time_values,
    c=mutual_information,  # Color by mutual information
    cmap='plasma_r',        # Colormap for better visualization
    edgecolor='k',
    alpha=0.7,
    s=100  # Adjust as needed
)

plt.colorbar(label='Mutual Information')  # Add colorbar for mutual information
plt.xlabel('Deviation (%)', fontsize=14)
plt.ylabel('Total Time (s)', fontsize=14)
plt.title('Scatter Plot of Deviation vs Total Time', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
