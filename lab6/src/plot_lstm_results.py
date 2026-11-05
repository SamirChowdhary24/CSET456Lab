import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

history_path = "lab6/output/lstm_training_history.csv"

df = pd.read_csv(history_path)

print("Training history loaded successfully.")
print(df.head())


# ============================================================
# GRAPH 1: TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    df["epoch"],
    df["validation_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "lab6/output/screenshots/lstm_loss_comparison.png",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 2: VALIDATION PERFORMANCE
# ============================================================

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

plt.title("LSTM Validation Performance")
plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)

plt.savefig(
    "lab6/output/screenshots/lstm_accuracy_f1.png",
    bbox_inches="tight"
)

plt.show()

print("\nGraphs saved successfully:")
print("- lab6/output/screenshots/lstm_loss_comparison.png")
print("- lab6/output/screenshots/lstm_accuracy_f1.png")