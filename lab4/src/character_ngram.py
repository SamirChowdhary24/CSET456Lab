
from pathlib import Path

import numpy as np
import pandas as pd


# Resolve paths relative to this script.
LAB4_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = LAB4_DIR / "data"
OUTPUT_DIR = LAB4_DIR / "output"


def get_trigrams(token):
    """Extract overlapping character trigrams from a token."""
    token = token.lower()
    return [token[i:i + 3] for i in range(len(token) - 2)]


def main():
    # Load the same 20 tokens for both embedding approaches.
    df = pd.read_csv(DATA_DIR / "selected_tokens.csv")
    tokens = df["token"].tolist()

    # Build a sorted trigram vocabulary.
    trigram_vocab = sorted({
        trigram
        for token in tokens
        for trigram in get_trigrams(token)
    })

    trigram_to_index = {
        trigram: i for i, trigram in enumerate(trigram_vocab)
    }

    # Each row represents one token.
    # Each column represents one character trigram.
    embeddings = np.zeros(
        (len(tokens), len(trigram_vocab)),
        dtype=np.float32
    )

    for row, token in enumerate(tokens):
        for trigram in get_trigrams(token):
            col = trigram_to_index[trigram]
            embeddings[row, col] += 1

    # Normalize vectors and calculate cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T

    # Calculate similarities for every unique token pair.
    pairs = []

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            pairs.append({
                "token_1": tokens[i],
                "token_2": tokens[j],
                "cosine_similarity": float(similarity[i, j])
            })

    pair_df = pd.DataFrame(pairs)
    pair_df = pair_df.sort_values(
        "cosine_similarity", ascending=False
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save embeddings, vocabulary and similarity results.
    np.save(OUTPUT_DIR / "character_ngram_embeddings.npy", embeddings)

    pd.DataFrame({"trigram": trigram_vocab}).to_csv(
        OUTPUT_DIR / "character_ngram_vocabulary.csv",
        index=False
    )

    pair_df.to_csv(
        OUTPUT_DIR / "character_ngram_pairwise_similarity.csv",
        index=False
    )

    pair_df.head(5).to_csv(
        OUTPUT_DIR / "character_ngram_top5_pairs.csv",
        index=False
    )

    print("Tokens:", len(tokens))
    print("Trigram vocabulary size:", len(trigram_vocab))
    print("Embedding matrix shape:", embeddings.shape)
    print("Unique token pairs:", len(pair_df))
    print("\nTop 5 most similar pairs:")
    print(pair_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
