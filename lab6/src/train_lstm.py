import time
import random
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

MAX_VOCAB_SIZE = 20_000
MAX_SEQUENCE_LENGTH = 300

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

DATA_PATH = "lab6/data/raw/phishing_email.csv"

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original dataset size: {len(df)}")

# Remove empty or whitespace-only emails.
df["text_combined"] = df["text_combined"].fillna("").astype(str)
df = df[df["text_combined"].str.strip() != ""].copy()

print(f"Dataset size after removing empty emails: {len(df)}")

texts = df["text_combined"].tolist()
labels = df["label"].astype(int).to_numpy()


# ============================================================
# CREATE TRAIN / VALIDATION / TEST SPLITS
# ============================================================

# First: 80% training and 20% temporary test data.
train_texts, test_texts, train_labels, test_labels = (
    train_test_split(
        texts,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=RANDOM_SEED
    )
)

# Split the training portion again to create validation data.
train_texts, val_texts, train_labels, val_labels = (
    train_test_split(
        train_texts,
        train_labels,
        test_size=0.10,
        stratify=train_labels,
        random_state=RANDOM_SEED
    )
)

print("\nDataset split:")
print(f"Training samples:   {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Testing samples:    {len(test_texts)}")


# ============================================================
# BUILD VOCABULARY
# ============================================================

print("\nBuilding vocabulary from training data...")

word_counter = Counter()

for text in train_texts:
    word_counter.update(text.lower().split())

# Reserve 0 for padding and 1 for unknown words.
vocabulary = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, _ in word_counter.most_common(MAX_VOCAB_SIZE - 2):
    vocabulary[word] = len(vocabulary)

print(f"Vocabulary size: {len(vocabulary)}")


# ============================================================
# TEXT TO NUMERICAL SEQUENCES
# ============================================================

def text_to_sequence(text):
    """Convert text into a fixed-length sequence of word IDs."""

    words = text.lower().split()

    sequence = [
        vocabulary.get(word, vocabulary["<UNK>"])
        for word in words[:MAX_SEQUENCE_LENGTH]
    ]

    # Pad shorter sequences with zeros.
    if len(sequence) < MAX_SEQUENCE_LENGTH:
        sequence.extend(
            [vocabulary["<PAD>"]]
            * (MAX_SEQUENCE_LENGTH - len(sequence))
        )

    return sequence


print("Converting text into numerical sequences...")

X_train = np.array(
    [text_to_sequence(text) for text in train_texts],
    dtype=np.int64
)

X_val = np.array(
    [text_to_sequence(text) for text in val_texts],
    dtype=np.int64
)

X_test = np.array(
    [text_to_sequence(text) for text in test_texts],
    dtype=np.int64
)


# ============================================================
# CREATE PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.long),
    torch.tensor(train_labels, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.long),
    torch.tensor(val_labels, dtype=torch.float32)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.long),
    torch.tensor(test_labels, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LSTM MODEL
# ============================================================

class PhishingLSTM(nn.Module):
    """LSTM model for phishing email classification."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x):

        embedded = self.embedding(x)

        # LSTM returns outputs for every time step.
        # We use the final hidden state for classification.
        _, (hidden, _) = self.lstm(embedded)

        output = hidden[-1]

        return self.classifier(output).squeeze(1)


model = PhishingLSTM(
    vocab_size=len(vocabulary),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

print("\nModel architecture:")
print(model)


# ============================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def evaluate(model, data_loader):
    """Evaluate loss and classification metrics."""

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for features, labels_batch in data_loader:

            features = features.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels_batch
            )

            total_loss += (
                loss.item() * labels_batch.size(0)
            )

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= 0.5
            ).long()

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

    average_loss = (
        total_loss / len(data_loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="binary",
            zero_division=0
        )
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting LSTM training...")

history = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for features, labels_batch in train_loader:

        features = features.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels_batch.size(0)
        )

    train_loss = (
        running_loss / len(train_loader.dataset)
    )

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = evaluate(model, val_loader)

    elapsed_time = time.time() - start_time

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "validation_f1": val_f1,
        "elapsed_seconds": elapsed_time
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

total_training_time = time.time() - start_time

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1
) = evaluate(model, test_loader)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1 Score:  {test_f1:.4f}")
print(f"Training Time:  {total_training_time:.2f} seconds")


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR = "lab6/output"

history_df = pd.DataFrame(history)

history_path = (
    f"{OUTPUT_DIR}/lstm_training_history.csv"
)

results_path = (
    f"{OUTPUT_DIR}/lstm_final_results.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

results_df = pd.DataFrame([{
    "model": "LSTM",
    "dataset": "Phishing Email Dataset",
    "total_samples": len(df),
    "training_samples": len(train_texts),
    "validation_samples": len(val_texts),
    "test_samples": len(test_texts),
    "vocabulary_size": len(vocabulary),
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "embedding_dimension": EMBEDDING_DIM,
    "hidden_dimension": HIDDEN_DIM,
    "epochs": EPOCHS,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1": test_f1,
    "training_time_seconds": total_training_time
}])

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved:")
print(f"- {history_path}")
print(f"- {results_path}")