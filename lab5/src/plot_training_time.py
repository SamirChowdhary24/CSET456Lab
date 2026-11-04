import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# LOAD MODEL COMPARISON RESULTS
# ============================================================

comparison_path = Path("lab5/output/model_comparison_full.csv")

df = pd.read_csv(comparison_path)

print("Model comparison data:")
print(df[[
    "model",
    "architecture",
    "total_training_seconds"
]].to_string(index=False))


# ============================================================
# CREATE TRAINING TIME COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["architecture"],
    df["total_training_seconds"]
)

plt.title("Training Time Comparison Across Neural Network Architectures")
plt.xlabel("Hidden Layer Architecture")
plt.ylabel("Total Training Time (seconds)")
plt.grid(axis="y")


# ============================================================
# SAVE GRAPH
# ============================================================

output_path = Path("lab5/output/training_time_comparison.png")

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nGraph saved to: {output_path}")

plt.show()