import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['CITY1', 'CITY2', 'ROCKY', 'DESERT', 'AMAZON']
day = np.array([0.362, 0.219, 1.909, 0.251, 0.381])
early_eve = np.array([0.363, 0.222, 2.071, 0.425, 0.578])
late_eve = np.array([0.365, 0.225, 2.290, 28.713, 141.647])
midnight = np.array([0.371, 0.310, 20.399, 112.610, 153.678])

# Calculate percentage increase from baseline (Day)
early_eve_increase = ((early_eve - day) / day) * 100
late_eve_increase = ((late_eve - day) / day) * 100
midnight_increase = ((midnight - day) / day) * 100

# Bar width and positioning
bar_width = 0.3
index = np.arange(len(datasets))

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(index, early_eve_increase, bar_width, label='Early Eve', color='lightgreen')
ax.bar(index + bar_width, late_eve_increase, bar_width, label='Late Eve', color='gold')
ax.bar(index + 2 * bar_width, midnight_increase, bar_width, label='Midnight', color='lightcoral')

# Labels and title
ax.set_xlabel('Dataset', fontsize=20)
ax.set_ylabel('Percentage Increase from Day (%) (Log Scale)', fontsize=20)
ax.set_title('Percentage Increase in Mean Radial Percent for Different Lighting Conditions', fontsize=23)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(datasets, fontsize=12)

# Legend and layout adjustments
ax.legend(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('symlog')
plt.xticks(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
