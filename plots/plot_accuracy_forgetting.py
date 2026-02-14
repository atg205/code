import matplotlib.pyplot as plt
from pathlib import Path

# Poster font sizes
TITLE_FONTSIZE = 48
LABEL_FONTSIZE = 40
TICK_FONTSIZE = 36
LEGEND_FONTSIZE = 38

# X values (iterations)
x = [0, 1, 2, 3, 4]

# Top plot — Mean Accuracy
mean_accuracy_random = [0.79, 0.66, 0.60, 0.58, 0.55]
mean_accuracy_sorted = [0.79, 0.67, 0.60, 0.57, 0.53]

# Bottom plot — Forgetting Score
# (values shown from iteration 1 to 4 in the figure)
x_forgetting = [1, 2, 3, 4]

forgetting_random = [0.158, 0.148, 0.156, 0.165]
forgetting_sorted = [0.142, 0.165, 0.198, 0.230]

# Create side-by-side plots with shared legend
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Mean Accuracy plot
ax1.scatter(x, mean_accuracy_random, label='Aléatoire', marker='x', s=800, linewidths=4, color='#d62728')
ax1.plot(x, mean_accuracy_random, color='#d62728', linewidth=2, alpha=0.5)
ax1.scatter(x, mean_accuracy_sorted, label='Trié', marker='o', s=800, color='#2ca02c')
ax1.plot(x, mean_accuracy_sorted, color='#2ca02c', linewidth=2, alpha=0.5)
ax1.set_xlabel('Tâche', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax1.set_ylabel('Précision Moyenne', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax1.set_xticks(x)
ax1.tick_params(axis='both', labelsize=TICK_FONTSIZE)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.5, 0.85)

# Forgetting Score plot
ax2.scatter(x_forgetting, forgetting_random, label='Aléatoire', marker='x', s=800, linewidths=4, color='#d62728')
ax2.plot(x_forgetting, forgetting_random, color='#d62728', linewidth=2, alpha=0.5)
ax2.scatter(x_forgetting, forgetting_sorted, label='Trié', marker='o', s=800, color='#2ca02c')
ax2.plot(x_forgetting, forgetting_sorted, color='#2ca02c', linewidth=2, alpha=0.5)
ax2.set_xlabel('Tâche', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax2.set_ylabel('Score d\'Oubli', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax2.set_xticks(x_forgetting)
ax2.tick_params(axis='both', labelsize=TICK_FONTSIZE)
ax2.grid(True, alpha=0.3)

# Shared legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, framealpha=0.95)

base = Path(__file__).resolve().parent
out_path = base / 'accuracy_forgetting_comparison.png'

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
plt.show()
