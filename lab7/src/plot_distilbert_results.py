import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "lab7/output/distilbert_training_history.csv"
OUTPUT_DIR = "lab7/output/screenshots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

history = pd.read_csv(INPUT_PATH)

print("Training history loaded successfully.")
print(history)


# ============================================================
# PLOT 1: TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["validation_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("DistilBERT Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(history["epoch"])
plt.grid(True)
plt.legend()

loss_path = (
    f"{OUTPUT_DIR}/distilbert_loss_comparison.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {loss_path}")


# ============================================================
# PLOT 2: VALIDATION ACCURACY AND F1 SCORE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["validation_accuracy"] * 100,
    marker="o",
    label="Validation Accuracy"
)

plt.plot(
    history["epoch"],
    history["validation_f1"] * 100,
    marker="o",
    label="Validation F1 Score"
)

plt.title("DistilBERT Validation Accuracy and F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.xticks(history["epoch"])
plt.grid(True)
plt.legend()

performance_path = (
    f"{OUTPUT_DIR}/distilbert_accuracy_f1.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {performance_path}")