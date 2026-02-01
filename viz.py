import json
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load results
# -----------------------------
with open("xgb_iteration_results_20260125_171701.json", "r") as f:
    xgb_results = json.load(f)

with open("iteration_results_20260116_194542.json", "r") as f:
    tramel_results = json.load(f)

# -----------------------------
# Extract XGBoost metrics
# -----------------------------
xgb_iters = [r["iteration"] for r in xgb_results]
xgb_mean_acc = [r["mean_cv_score"] for r in xgb_results]
xgb_time = [r["time_seconds"] for r in xgb_results]
xgb_per_class = [r["per_class_acc"] for r in xgb_results]

# -----------------------------
# Extract Tramel metrics
# -----------------------------
tramel_iters = [r["iteration"] for r in tramel_results]
tramel_mean_acc = [np.mean(r["task_accuracies"]) for r in tramel_results]
tramel_time = [r["time_seconds"] for r in tramel_results]
tramel_per_task = [r["task_accuracies"] for r in tramel_results]

# -----------------------------
# Global matplotlib style
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 120
})

# ============================================================
# Figure 1 — Mean accuracy vs iteration
# ============================================================
plt.figure(figsize=(6.5, 4))

plt.plot(xgb_iters, xgb_mean_acc, marker="o", linewidth=2, label="XGBoost")
plt.plot(tramel_iters, tramel_mean_acc, marker="s", linewidth=2, label="Tramel Network")

plt.xlabel("Incremental iteration")
plt.ylabel("Mean accuracy")
plt.title("Overall accuracy across incremental tasks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_mean_accuracy.pdf")
# ============================================================
# Figure 2 — Per-class / per-task accuracy evolution
# (color = iteration, marker/line = method)
# ============================================================

from matplotlib.lines import Line2D

plt.figure(figsize=(7.4, 4.6))

colors = plt.cm.tab10.colors
max_iters = max(len(xgb_per_class), len(tramel_per_task))

# Plot curves
for i in range(max_iters):
    color = colors[i % len(colors)]

    # XGBoost
    if i < len(xgb_per_class):
        accs = xgb_per_class[i]
        plt.plot(
            range(len(accs)),
            accs,
            color=color,
            linestyle="-",
            marker="o",
            markersize=7,
            linewidth=2,
            alpha=0.9
        )

    # Tramel Network
    if i < len(tramel_per_task):
        accs = tramel_per_task[i]
        plt.plot(
            range(len(accs)),
            accs,
            color=color,
            linestyle="--",
            marker="s",
            markersize=7,
            linewidth=2,
            alpha=0.9
        )

plt.xlabel("Class / task index")
plt.ylabel("Accuracy")
plt.title("Per-class / per-task accuracy as tasks accumulate")
plt.grid(True, alpha=0.3)

# -----------------------------
# Legend 1 — Methods
# -----------------------------
method_legend = [
    Line2D([0], [0], color="black", linestyle="-", marker="o",
           markersize=7, linewidth=2, label="XGBoost"),
    Line2D([0], [0], color="black", linestyle="--", marker="s",
           markersize=7, linewidth=2, label="Tramel Network")
]

legend_methods = plt.legend(
    handles=method_legend,
    loc="lower left",
    frameon=False
)

# -----------------------------
# Legend 2 — Iterations (colors)
# -----------------------------
iteration_legend = [
    Line2D([0], [0], color=colors[i % len(colors)],
           linewidth=3, label=f"Iteration {i}")
    for i in range(max_iters)
]

legend_iters = plt.legend(
    handles=iteration_legend,
    title="Incremental iteration",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

# Ensure both legends are drawn
plt.gca().add_artist(legend_methods)

plt.tight_layout()
plt.savefig("comparison_per_class_accuracy.pdf", bbox_inches="tight")
plt.show()



# ============================================================
# Figure 3 — Training time per iteration
# ============================================================
plt.figure(figsize=(6.5, 4))

plt.plot(xgb_iters, xgb_time, marker="o", linewidth=2, label="XGBoost")
plt.plot(tramel_iters, tramel_time, marker="s", linewidth=2, label="Tramel Network")

plt.xlabel("Incremental iteration")
plt.ylabel("Training time (seconds)")
plt.title("Training time per incremental iteration")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_training_time.pdf")

# -----------------------------
# Show all figures nicely
# -----------------------------
plt.show()

print("Figures saved:")
print("- comparison_mean_accuracy.pdf")
print("- comparison_per_class_accuracy.pdf")
print("- comparison_training_time.pdf")
