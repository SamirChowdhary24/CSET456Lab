import os
import numpy as np
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CORPUS_FILE = "lab2/data/processed/text_corpus.csv"
TOKENIZER_FILE = "lab3/data/subword_bpe_tokenizer.json"
OUTPUT_DIR = "lab3/output"

EMBEDDING_DIM = 100
TOP_K = 50
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading corpus and tokenizer...")

df = pd.read_csv(CORPUS_FILE)
texts = df["text"].fillna("").astype(str).tolist()

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

# ---------------------------------------------------------
# Count subword token frequencies
# ---------------------------------------------------------

print("Counting subword frequencies...")

token_counter = Counter()

for text in texts:
    encoding = tokenizer.encode(text)
    token_counter.update(encoding.tokens)

top_tokens = token_counter.most_common(TOP_K)

print("\nTop 50 tokens:")
for rank, (token, frequency) in enumerate(top_tokens, start=1):
    print(f"{rank:2}. {repr(token):25} {frequency}")

# ---------------------------------------------------------
# Create random embedding matrix
# ---------------------------------------------------------

print("\nCreating random embedding matrix...")

rng = np.random.default_rng(RANDOM_SEED)

tokens = [token for token, _ in top_tokens]

embeddings = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(TOP_K, EMBEDDING_DIM)
)

# Normalize embeddings for cosine similarity
normalized = embeddings / np.linalg.norm(
    embeddings, axis=1, keepdims=True
)

similarity_matrix = normalized @ normalized.T

# ---------------------------------------------------------
# Average similarity of each token
# ---------------------------------------------------------

average_similarities = []

for i, token in enumerate(tokens):
    similarities = np.delete(similarity_matrix[i], i)
    average_similarities.append(
        similarities.mean()
    )

similarity_results = pd.DataFrame({
    "token": tokens,
    "frequency": [freq for _, freq in top_tokens],
    "average_cosine_similarity": average_similarities
})

# Most similar
most_similar = similarity_results.sort_values(
    "average_cosine_similarity",
    ascending=False
).head(10)

# Least similar
least_similar = similarity_results.sort_values(
    "average_cosine_similarity",
    ascending=True
).head(10)

print("\nTop 10 most similar embeddings:")
print(most_similar.to_string(index=False))

print("\nTop 10 least similar embeddings:")
print(least_similar.to_string(index=False))

# ---------------------------------------------------------
# Select 5 tokens from top 50
# ---------------------------------------------------------

selected_indices = [0, 10, 20, 30, 40]

selected_tokens = [
    tokens[i] for i in selected_indices
]

print("\nSelected 5 tokens:")
for token in selected_tokens:
    print(repr(token))

# ---------------------------------------------------------
# Find nearest tokens for each selected token
# ---------------------------------------------------------

nearest_results = []

for token in selected_tokens:

    token_index = tokens.index(token)

    similarities = similarity_matrix[token_index].copy()

    # Exclude itself
    similarities[token_index] = -np.inf

    nearest_indices = np.argsort(
        similarities
    )[::-1][:5]

    for rank, idx in enumerate(nearest_indices, start=1):

        nearest_results.append({
            "selected_token": token,
            "nearest_rank": rank,
            "nearest_token": tokens[idx],
            "cosine_similarity": similarities[idx]
        })

nearest_df = pd.DataFrame(nearest_results)

print("\nNearest tokens:")
print(nearest_df.to_string(index=False))

# ---------------------------------------------------------
# Pairwise similarity between selected 5 tokens
# ---------------------------------------------------------

pair_results = []

for i in range(len(selected_tokens)):
    for j in range(i + 1, len(selected_tokens)):

        token_a = selected_tokens[i]
        token_b = selected_tokens[j]

        index_a = tokens.index(token_a)
        index_b = tokens.index(token_b)

        similarity = similarity_matrix[index_a, index_b]

        pair_results.append({
            "token_a": token_a,
            "token_b": token_b,
            "cosine_similarity": similarity
        })

pair_df = pd.DataFrame(pair_results)

print("\nPairwise similarity:")
print(pair_df.to_string(index=False))

# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

similarity_results.to_csv(
    f"{OUTPUT_DIR}/top50_embedding_similarity.csv",
    index=False
)

most_similar.to_csv(
    f"{OUTPUT_DIR}/top10_most_similar.csv",
    index=False
)

least_similar.to_csv(
    f"{OUTPUT_DIR}/top10_least_similar.csv",
    index=False
)

nearest_df.to_csv(
    f"{OUTPUT_DIR}/selected_token_nearest_neighbors.csv",
    index=False
)

pair_df.to_csv(
    f"{OUTPUT_DIR}/selected_token_pair_similarity.csv",
    index=False
)

np.save(
    f"{OUTPUT_DIR}/random_embedding_matrix.npy",
    embeddings
)

print("\n" + "=" * 60)
print("EXERCISES 2, 3 AND 4 COMPLETED")
print("=" * 60)
print("Results saved in:", OUTPUT_DIR)