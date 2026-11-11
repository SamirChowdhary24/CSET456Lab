import time
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

MODEL_NAME = "distilbert-base-uncased"

# Reduced sequence length for faster CPU training.
MAX_LENGTH = 128

# Batch size for CPU training.
BATCH_SIZE = 16

LEARNING_RATE = 2e-5
EPOCHS = 3

# Use a smaller representative subset of the training data because
# full DistilBERT fine-tuning is computationally expensive on CPU.
TRAINING_SUBSET_SIZE = 5_000

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
# Same split logic as the LSTM experiment.
# ============================================================

train_texts, test_texts, train_labels, test_labels = (
    train_test_split(
        texts,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=RANDOM_SEED
    )
)

train_texts, val_texts, train_labels, val_labels = (
    train_test_split(
        train_texts,
        train_labels,
        test_size=0.10,
        stratify=train_labels,
        random_state=RANDOM_SEED
    )
)

print("\nOriginal dataset split:")
print(f"Training samples:   {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Testing samples:    {len(test_texts)}")


# ============================================================
# CREATE A STRATIFIED TRAINING SUBSET
# ============================================================

if len(train_texts) > TRAINING_SUBSET_SIZE:

    train_texts, _, train_labels, _ = (
        train_test_split(
            train_texts,
            train_labels,
            train_size=TRAINING_SUBSET_SIZE,
            stratify=train_labels,
            random_state=RANDOM_SEED
        )
    )

print("\nDistilBERT training setup:")
print(f"Training subset:    {len(train_texts)} samples")
print(f"Validation samples: {len(val_texts)} samples")
print(f"Testing samples:    {len(test_texts)} samples")


# ============================================================
# LOAD DISTILBERT TOKENIZER
# ============================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKENIZE TEXT
# ============================================================

print("Tokenizing training data...")

train_encoding = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

print("Tokenizing validation data...")

val_encoding = tokenizer(
    val_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

print("Tokenizing test data...")

test_encoding = tokenizer(
    test_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)


# ============================================================
# CREATE PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    train_encoding["input_ids"],
    train_encoding["attention_mask"],
    torch.tensor(train_labels, dtype=torch.long)
)

val_dataset = TensorDataset(
    val_encoding["input_ids"],
    val_encoding["attention_mask"],
    torch.tensor(val_labels, dtype=torch.long)
)

test_dataset = TensorDataset(
    test_encoding["input_ids"],
    test_encoding["attention_mask"],
    torch.tensor(test_labels, dtype=torch.long)
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
# LOAD PRE-TRAINED DISTILBERT MODEL
# ============================================================

print("\nLoading pre-trained DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(DEVICE)

model_parameters = sum(
    p.numel() for p in model.parameters()
)

print(f"Model parameters: {model_parameters:,}")


# ============================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
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

        for input_ids, attention_mask, labels_batch in data_loader:

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                outputs.logits,
                labels_batch
            )

            total_loss += (
                loss.item() * labels_batch.size(0)
            )

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

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
# TRAIN DISTILBERT
# ============================================================

print("\nStarting DistilBERT fine-tuning...")
print(
    f"Training batches per epoch: "
    f"{len(train_loader)}"
)

history = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for input_ids, attention_mask, labels_batch in train_loader:

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(
            outputs.logits,
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
    ) = evaluate(
        model,
        val_loader
    )

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
        f"Val F1: {val_f1:.4f} | "
        f"Time: {elapsed_time:.2f}s"
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
) = evaluate(
    model,
    test_loader
)

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

OUTPUT_DIR = "lab7/output"

history_df = pd.DataFrame(history)

history_path = (
    f"{OUTPUT_DIR}/distilbert_training_history.csv"
)

results_path = (
    f"{OUTPUT_DIR}/distilbert_final_results.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

results_df = pd.DataFrame([{
    "model": "DistilBERT",
    "dataset": "Phishing Email Dataset",
    "total_samples": len(df),
    "training_samples": len(train_texts),
    "validation_samples": len(val_texts),
    "test_samples": len(test_texts),
    "max_sequence_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "model_parameters": model_parameters,
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