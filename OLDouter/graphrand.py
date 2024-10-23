import numpy as np
import matplotlib.pyplot as plt

# Define delta1 range from -180 to 180
delta1_range = np.linspace(-180, 180, 500)
# Initialize lists for losses and valid delta1, delta2 values
delta1_vals, delta2_vals = [], []
pre_norm_loss, post_norm_loss = [], []

# Iterate over delta1 and delta2 values, compute loss if constraint is met
for d1 in delta1_range:
    for d2 in np.linspace(-180, 180, 500):
        if -200 <= d2 - d1 <= -160:  # Constraint: delta2 = delta1 - (180 +/- 20)
            delta1_vals.append(d1)
            delta2_vals.append(d2)
            pre_norm_loss.append(np.abs(np.sin(np.radians(np.abs(d1))) + np.sin(np.radians(np.abs(d2)))))
            post_norm_loss.append(np.abs(np.sin(np.radians(np.abs(d2 - d1)))))

# Create the 3D plot for the point cloud
fig = plt.figure(figsize=(12, 6))

# Pre-Normalization Loss
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(delta1_vals, delta2_vals, pre_norm_loss, c=pre_norm_loss, cmap='viridis')
ax1.set_title('Pre-Normalization Loss')
ax1.set_xlabel('Delta1 (degrees)')
ax1.set_ylabel('Delta2 (degrees)')
ax1.set_zlabel('Loss')

# Post-Normalization Loss
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(delta1_vals, delta2_vals, post_norm_loss, c=post_norm_loss, cmap='plasma')
ax2.set_title('Post-Normalization Loss')
ax2.set_xlabel('Delta1 (degrees)')
ax2.set_ylabel('Delta2 (degrees)')
ax2.set_zlabel('Loss')

plt.tight_layout()
plt.show()
