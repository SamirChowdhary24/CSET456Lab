import pandas as pd
import matplotlib.pyplot as plt


# Load training history
history_path = "lab6/output/lstm_training_history.csv"
df = pd.read_csv(history_path)

# Create validation performance graph
plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["validation_accuracy"] * 100,
    marker="o",
    label="Validation Accuracy"
)

plt.plot(
    df["epoch"],
    df["validation_f1"] * 100,
    marker="o",
    label="Validation F1 Score"
)

plt.title("LSTM Validation Accuracy and F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)

# Save the graph
output_path = "lab6/output/screenshots/lstm_accuracy_f1.png"

plt.savefig(
    output_path,
    bbox_inches="tight"
)

plt.show()

print(f"Graph saved successfully: {output_path}")