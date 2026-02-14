import matplotlib.pyplot as plt
import numpy as np

# Poster font sizes
TITLE_FONTSIZE = 48
LABEL_FONTSIZE = 40
TICK_FONTSIZE = 36
LEGEND_FONTSIZE = 38

# Data from the Stability-Plasticity Trade-off plot
# X: Average Forgetting Score, Y: Mean Accuracy
# Labels: a=value, b=value

data = [
    (0.12, 0.513, "a=4.0, b=0.0"),
    (0.18, 0.527, "a=4.0, b=1.0"),
    (0.19, 0.533, "a=1.0, b=0.0"),
    (0.24, 0.525, "a=0.0, b=0.0"),
    (0.25, 0.523, "a=4.0, b=4.0"),
    (0.30, 0.514, "a=1.0, b=4.0"),
    (0.35, 0.500, "a=0.0, b=4.0"),
]

# Extract x and y values
x_vals = [point[0] for point in data]
y_vals = [point[1] for point in data]
labels = [point[2] for point in data]

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Scatter plot
scatter = ax.scatter(x_vals, y_vals, s=600, alpha=0.7, color='#1f77b4', edgecolors='black', linewidth=2)

# Add labels to each point
for i, (x, y, label) in enumerate(data):
    ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=LABEL_FONTSIZE-8, fontweight='bold')

# Labels and formatting
ax.set_xlabel('Score d\'Oubli Moyen', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax.set_ylabel('Précision Moyenne', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax.set_title('Compromis Stabilité-Plasticité', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=20)

ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
ax.grid(True, alpha=0.3, linewidth=1.5)

# Set axis limits with some padding
ax.set_xlim(0.08, 0.38)
ax.set_ylim(0.495, 0.540)

plt.tight_layout()
plt.savefig('stability_plasticity_tradeoff.png', dpi=300, bbox_inches='tight')
plt.show()
