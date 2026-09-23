import re
import os
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "lab2/data/processed/text_corpus.csv"
OUTPUT_DIR = "lab3/output"
RESULT_FILE = "lab3/output/tokenizer_statistics.csv"

EMBEDDING_DIM = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load corpus
# ---------------------------------------------------------

print("Loading corpus...")

df = pd.read_csv(INPUT_FILE)
texts = df["text"].fillna("").astype(str).tolist()

print(f"Records loaded: {len(texts):,}")

# ---------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------

print("\nRunning character-level tokenizer...")

character_lengths = []
character_counter = Counter()

for text in texts:
    tokens = list(text)
    character_lengths.append(len(tokens))
    character_counter.update(tokens)

character_vocab_size = len(character_counter)
character_avg_length = sum(character_lengths) / len(character_lengths)

# ---------------------------------------------------------
# Word-level tokenizer
# ---------------------------------------------------------

print("Running word-level tokenizer...")

word_lengths = []
word_counter = Counter()

for text in texts:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    word_lengths.append(len(tokens))
    word_counter.update(tokens)

word_vocab_size = len(word_counter)
word_avg_length = sum(word_lengths) / len(word_lengths)

# ---------------------------------------------------------
# Subword-level tokenizer (BPE)
# ---------------------------------------------------------

print("Training subword BPE tokenizer...")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=10000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.save("lab3/data/subword_bpe_tokenizer.json")

subword_lengths = []
subword_counter = Counter()

for text in texts:
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens

    subword_lengths.append(len(tokens))
    subword_counter.update(tokens)

subword_vocab_size = tokenizer.get_vocab_size()
subword_avg_length = sum(subword_lengths) / len(subword_lengths)

# ---------------------------------------------------------
# Embedding matrix calculation
# ---------------------------------------------------------

def embedding_statistics(vocab_size):
    parameters = vocab_size * EMBEDDING_DIM
    memory_bytes = parameters * 4
    memory_mb = memory_bytes / (1024 ** 2)

    return parameters, memory_mb


results = []

for tokenizer_name, vocab_size, avg_length in [
    ("Character", character_vocab_size, character_avg_length),
    ("Word", word_vocab_size, word_avg_length),
    ("Subword-BPE", subword_vocab_size, subword_avg_length),
]:

    parameters, memory_mb = embedding_statistics(vocab_size)

    results.append({
        "tokenizer": tokenizer_name,
        "vocabulary_size": vocab_size,
        "average_sequence_length": round(avg_length, 2),
        "embedding_dimension": EMBEDDING_DIM,
        "embedding_parameters": parameters,
        "embedding_memory_MB": round(memory_mb, 4),
    })

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(RESULT_FILE, index=False)

print("\n" + "=" * 60)
print("TOKENIZER COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved to:")
print(RESULT_FILE)

print("\nExperiment 1 completed successfully.")