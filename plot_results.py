import json
import numpy as np
import matplotlib.pyplot as plt

# Poster font sizes
TITLE_FONTSIZE = 48
LABEL_FONTSIZE = 40
TICK_FONTSIZE = 36
LEGEND_FONTSIZE = 38

with open('iteration_results_20260116_194542.json','r') as tramel_results_file:
    tramel_results = json.load(tramel_results_file)

with open('xgb_iteration_results_20260118_180220.json','r') as xgb_results_file:
    xgb_results = json.load(xgb_results_file)

xgb_success = [entry['mean_cv_score'] for entry in xgb_results]
xgb_time = [entry['time_seconds'] for entry in xgb_results]

tramel_success = [np.mean(entry['task_accuracies']) for entry in tramel_results]
tramel_time = [entry['time_seconds'] for entry in tramel_results]

# Create side-by-side plots with shared legend
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Performance plot
ax1.scatter([i for i in range(len(xgb_success))], xgb_success, label='XGBoost', marker='x', s=800, linewidths=4)
ax1.scatter([i for i in range(len(xgb_success))], tramel_success, label='Tramel', marker='o', s=800)
ax1.set_xlabel('Task', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax1.set_xticks([i for i in range(len(xgb_success))])
ax1.tick_params(axis='both', labelsize=TICK_FONTSIZE)
ax1.grid(True, alpha=0.3)

# Time plot
ax2.scatter([i for i in range(len(xgb_time))], xgb_time, label='XGBoost', marker='x', s=800, linewidths=4)
ax2.scatter([i for i in range(len(tramel_time))], tramel_time, label='Tramel', marker='o', s=800)
ax2.set_xlabel('Task', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax2.set_ylabel('Train time (s)', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax2.set_yscale('log')
ax2.set_ylim(1,)
ax2.set_xticks([i for i in range(len(xgb_time))])
ax2.tick_params(axis='both', labelsize=TICK_FONTSIZE)
ax2.grid(True, alpha=0.3)

# Shared legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, framealpha=0.95)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('performance_time_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

