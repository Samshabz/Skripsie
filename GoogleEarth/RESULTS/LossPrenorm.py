import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define delta1 and delta2 ranges from -180 to 180
delta1_range = np.linspace(-180, 180, 500)
delta2_range = np.linspace(-180, 180, 500)

# Initialize 2D arrays for pre and post normalization loss
pre_norm_loss = np.zeros((len(delta1_range), len(delta2_range)))
post_norm_loss = np.zeros((len(delta1_range), len(delta2_range)))

# Iterate over delta1 and delta2 values, compute loss if constraint is met
for i, d1 in enumerate(delta1_range):
    for j, d2 in enumerate(delta2_range):
        if -200 <= d2 - d1 <= -160:  # Constraint: delta2 = delta1 - (180 +/- 20)
            pre_norm_loss[i, j] = np.abs(np.sin(np.radians(np.abs(d1))) + np.sin(np.radians(np.abs(d2))))
            post_norm_loss[i, j] = np.abs(np.sin(np.radians(np.abs(d2 - d1))))

# Plot heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pre-Normalization Loss Heatmap with inverted color spectrum
sns.heatmap(pre_norm_loss, cmap='viridis_r', ax=ax1, cbar_kws={'label': 'Pre-Norm Loss'})
ax1.set_title('Pre-Normalization Loss')
ax1.set_xlabel('Delta2 Index')
ax1.set_ylabel('Delta1 Index')

# Post-Normalization Loss Heatmap with inverted color spectrum
sns.heatmap(post_norm_loss, cmap='plasma_r', ax=ax2, cbar_kws={'label': 'Post-Norm Loss'})
ax2.set_title('Post-Normalization Loss')
ax2.set_xlabel('Delta2 Index')
ax2.set_ylabel('Delta1 Index')

plt.tight_layout()
plt.show()
