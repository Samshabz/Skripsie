import matplotlib.pyplot as plt

# Data for each dataset
datasets = ["CITY1", "CITY2", "ROCKY", "DESERT", "AMAZON"]

# Accuracy metrics
rmse_error_m = [55.09, 8.56, 19.15, 31.44, 30.37]
gps_deviation_pct = [5.09, 0.80, 2.13, 3.54, 4.75]

# Time metrics
mean_add_time = [0.555, 0.522, 0.444, 0.505, 0.532]
mean_param_inference_time = [1.224, 1.120, 1.052, 1.232, 1.257]
mean_location_inference_time = [1.612, 1.761, 1.882, 1.814, 1.788]

# Plot 1: Accuracy Metrics (RMSE and Percentage GPS Deviation)
plt.figure(figsize=(10, 6))
plt.plot(datasets, rmse_error_m, marker='o', label="RMSE GPS Error (m)")
plt.plot(datasets, gps_deviation_pct, marker='o', label="Percentage GPS Deviation (%)")
plt.xlabel("Datasets")
plt.ylabel("Error")
plt.title("Accuracy Metrics Across Datasets")
plt.grid(True)

# Adding labels near the end of each line for clarity
plt.text(datasets[-1], rmse_error_m[-1]+2, "RMSE GPS Error (m)", va='bottom', ha='right')
plt.text(datasets[-1], gps_deviation_pct[-1]+2, "Percentage GPS Deviation (%)", va='bottom', ha='right')

# Plot 2: Time Metrics (Mean Add, Parameter, and Location Inference Time)
plt.figure(figsize=(10, 6))
plt.plot(datasets, mean_add_time, marker='o', label="Mean Add Time (s)")
plt.plot(datasets, mean_param_inference_time, marker='o', label="Mean Parameter Inference Time (s)")
plt.plot(datasets, mean_location_inference_time, marker='o', label="Mean Location Inference Time (s)")
plt.xlabel("Datasets")
plt.ylabel("Time (s)")
plt.title("Runtime Metrics Across Datasets")
plt.grid(True)

# Adding labels near the end of each line for clarity
plt.text(datasets[-1], mean_add_time[-1]+0.05, "Mean Add Time (s)", va='bottom', ha='right')
plt.text(datasets[-1], mean_param_inference_time[-1]+0.05, "Mean Parameter Inference Time (s)", va='bottom', ha='right')
plt.text(datasets[-1], mean_location_inference_time[-1]+0.051, "Mean Location Inference Time (s)", va='bottom', ha='right')

plt.show()
