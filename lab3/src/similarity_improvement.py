import numpy as np
import pandas as pd
from tokenizers import Tokenizer

CORPUS_FILE = "lab2/data/processed/text_corpus.csv"
TOKENIZER_FILE = "lab3/data/subword_bpe_tokenizer.json"
EMBEDDING_FILE = "lab3/output/random_embedding_matrix.npy"
TOP50_FILE = "lab3/output/top50_embedding_similarity.csv"
OUTPUT_DIR = "lab3/output"

WINDOW_SIZE = 2
ITERATIONS = 5
ALPHA = 0.2

print("Loading corpus, tokenizer and embeddings...")

df = pd.read_csv(CORPUS_FILE)
texts = df["text"].fillna("").astype(str).tolist()

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
embeddings = np.load(EMBEDDING_FILE)

top50_df = pd.read_csv(TOP50_FILE)
tokens = top50_df["token"].tolist()

token_to_index = {
    token: index for index, token in enumerate(tokens)
}

print("Building co-occurrence matrix...")

cooccurrence = np.zeros(
    (len(tokens), len(tokens)),
    dtype=np.float64
)

for text in texts:
    encoded = tokenizer.encode(text)

    token_ids = [
        token_to_index[token]
        for token in encoded.tokens
        if token in token_to_index
    ]

    for i, center in enumerate(token_ids):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(token_ids), i + WINDOW_SIZE + 1)

        for j in range(start, end):
            if i != j:
                context = token_ids[j]
                cooccurrence[center, context] += 1

print("Co-occurrence matrix created.")

row_sums = cooccurrence.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1

weights = cooccurrence / row_sums

improved_embeddings = embeddings.copy()

print("Improving embeddings...")

for iteration in range(ITERATIONS):
    neighbor_embeddings = weights @ improved_embeddings

    improved_embeddings = (
        (1 - ALPHA) * improved_embeddings
        + ALPHA * neighbor_embeddings
    )

    norms = np.linalg.norm(
        improved_embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    improved_embeddings = improved_embeddings / norms

    print(f"Iteration {iteration + 1}/{ITERATIONS} completed.")


def cosine_matrix(matrix):
    normalized = matrix / np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    return normalized @ normalized.T


before_similarity = cosine_matrix(embeddings)
after_similarity = cosine_matrix(improved_embeddings)

results = []

for i, token_a in enumerate(tokens):
    for j in range(i + 1, len(tokens)):

        token_b = tokens[j]

        results.append({
            "token_a": token_a,
            "token_b": token_b,
            "similarity_before": before_similarity[i, j],
            "similarity_after": after_similarity[i, j],
            "change": (
                after_similarity[i, j]
                - before_similarity[i, j]
            )
        })

comparison_df = pd.DataFrame(results)

average_before = comparison_df["similarity_before"].mean()
average_after = comparison_df["similarity_after"].mean()

print("\n" + "=" * 60)
print("SIMILARITY IMPROVEMENT RESULTS")
print("=" * 60)

print(f"Average similarity before: {average_before:.6f}")
print(f"Average similarity after:  {average_after:.6f}")
print(
    f"Change:                    "
    f"{average_after - average_before:.6f}"
)

largest_improvements = comparison_df.sort_values(
    "change",
    ascending=False
).head(10)

print("\nTop 10 largest similarity improvements:")
print(largest_improvements.to_string(index=False))

np.save(
    f"{OUTPUT_DIR}/improved_embedding_matrix.npy",
    improved_embeddings
)

comparison_df.to_csv(
    f"{OUTPUT_DIR}/similarity_before_after.csv",
    index=False
)

largest_improvements.to_csv(
    f"{OUTPUT_DIR}/top_similarity_improvements.csv",
    index=False
)

print("\nResults saved in:", OUTPUT_DIR)
print("Exercise 5 and Exercise 6 completed.")