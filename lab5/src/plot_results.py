import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load training history
history_path = Path("lab5/output/training_history_full.csv")
df = pd.read_csv(history_path)


# Create the accuracy comparison graph
plt.figure(figsize=(10, 6))

for model_name, group in df.groupby("model"):
    plt.plot(
        group["epoch"],
        group["test_accuracy"] * 100,
        marker="o",
        label=model_name
    )


plt.title("Test Accuracy Comparison Across Neural Network Architectures")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.legend()
plt.grid(True)

# Save the graph
output_path = Path("lab5/output/accuracy_comparison_full.png")

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Graph saved to: {output_path}")

plt.show()