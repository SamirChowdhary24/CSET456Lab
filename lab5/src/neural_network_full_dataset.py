import time
import random

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
TEST_SIZE = 0.20

BATCH_SIZE = 512
LEARNING_RATE = 0.001
MAX_EPOCHS = 30

# A higher target makes it easier to compare learning speed.
TARGET_ACCURACY = 0.80

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """Set random seeds so the experiment is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD FULL DATASET
# ============================================================

print("\nLoading Forest Cover Type dataset...")

X, y = fetch_covtype(return_X_y=True)

print(f"Full dataset shape: {X.shape}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# PREPARE PYTORCH DATA
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Forest Cover labels are 1-7, but CrossEntropyLoss expects 0-6.
y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long)
y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# NEURAL NETWORK MODEL
# ============================================================

class NeuralNetwork(nn.Module):
    """Create a neural network from a list of hidden layer sizes."""

    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, data_loader):
    """Calculate classification accuracy."""

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in data_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model_name, hidden_layers):

    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print(f"Hidden layers: {hidden_layers}")
    print("=" * 60)

    # Reset the seed so each experiment starts consistently.
    set_seed(RANDOM_SEED)

    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_size=7
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []
    epoch_target_reached = None

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        running_loss = 0.0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        average_loss = running_loss / len(train_loader.dataset)

        test_accuracy = evaluate_model(model, test_loader)

        elapsed_time = time.time() - start_time

        if (
            epoch_target_reached is None
            and test_accuracy >= TARGET_ACCURACY
        ):
            epoch_target_reached = epoch

        history.append({
            "model": model_name,
            "architecture": "-".join(map(str, hidden_layers)),
            "epoch": epoch,
            "training_loss": average_loss,
            "test_accuracy": test_accuracy,
            "elapsed_seconds": elapsed_time
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

    total_time = time.time() - start_time
    final_accuracy = history[-1]["test_accuracy"]

    print(f"\nFinished {model_name}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Total training time: {total_time:.2f} seconds")

    if epoch_target_reached is not None:
        print(
            f"Target accuracy ({TARGET_ACCURACY:.0%}) "
            f"reached at epoch: {epoch_target_reached}"
        )
    else:
        print(
            f"Target accuracy ({TARGET_ACCURACY:.0%}) "
            "was not reached."
        )

    return history, {
        "model": model_name,
        "architecture": " → ".join(map(str, hidden_layers)),
        "hidden_layers": len(hidden_layers),
        "total_hidden_nodes": sum(hidden_layers),
        "final_accuracy": final_accuracy,
        "epochs_to_target_accuracy": epoch_target_reached,
        "total_training_seconds": total_time
    }


# ============================================================
# RUN ALL ARCHITECTURES
# ============================================================

experiments = {
    "Model_1_30": [30],
    "Model_2_15_15": [15, 15],
    "Model_3_10_10_10": [10, 10, 10]
}

all_history = []
summary_results = []

for model_name, hidden_layers in experiments.items():

    history, summary = train_model(
        model_name,
        hidden_layers
    )

    all_history.extend(history)
    summary_results.append(summary)


# ============================================================
# SAVE FULL DATASET RESULTS
# ============================================================

output_directory = "lab5/output"

history_df = pd.DataFrame(all_history)
summary_df = pd.DataFrame(summary_results)

history_path = f"{output_directory}/training_history_full.csv"
summary_path = f"{output_directory}/model_comparison_full.csv"

history_df.to_csv(history_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("FULL DATASET EXPERIMENT COMPLETE")
print("=" * 60)

print("\nModel Comparison:")
print(summary_df.to_string(index=False))

print(f"\nTraining history saved to: {history_path}")
print(f"Model comparison saved to: {summary_path}")