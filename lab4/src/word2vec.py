
from pathlib import Path
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
LAB4_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = LAB4_DIR.parent

CORPUS_PATH = PROJECT_DIR / "lab2" / "data" / "processed" / "text_corpus.csv"
TOKENS_PATH = LAB4_DIR / "data" / "selected_tokens.csv"
OUTPUT_DIR = LAB4_DIR / "output" / "word2vec"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible Word2Vec configuration
VECTOR_SIZE = 50
WINDOW = 5
MIN_COUNT = 1
SEED = 42
EPOCHS = 10

def tokenize_text(text):
    """Extract identifier-like words and programming symbols."""
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\w\s]", str(text))

def main():
    print("Loading corpus...")
    corpus = pd.read_csv(CORPUS_PATH, usecols=["text"], dtype=str).fillna("")
    sentences = [
        tokens
        for text in corpus["text"]
        if (tokens := tokenize_text(text))
    ]
    print(f"Corpus records: {len(corpus):,}")
    print(f"Non-empty token sequences: {len(sentences):,}")

    print("Training Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=1,
        sg=1,
        seed=SEED,
        epochs=EPOCHS,
    )
    model.save(str(OUTPUT_DIR / "word2vec.model"))
    print(f"Vocabulary size: {len(model.wv):,}")

    selected = pd.read_csv(TOKENS_PATH)["token"].astype(str).tolist()
    available = [token for token in selected if token in model.wv]
    missing = [token for token in selected if token not in model.wv]
    print(f"Selected tokens found: {len(available)}/{len(selected)}")
    if missing:
        print("Missing tokens:", ", ".join(missing))

    if len(available) < 2:
        raise RuntimeError("Fewer than two selected tokens were found in Word2Vec vocabulary.")

    vectors = np.vstack([model.wv[token] for token in available])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    np.save(OUTPUT_DIR / "word2vec_embeddings.npy", vectors)

    pd.DataFrame({
        "token": available,
        "embedding": [vector.tolist() for vector in vectors],
    }).to_json(OUTPUT_DIR / "word2vec_embedding_table.json", orient="records", indent=2)

    similarities = cosine_similarity(normalized)
    rows = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            rows.append({
                "token_1": available[i],
                "token_2": available[j],
                "cosine_similarity": float(similarities[i, j]),
            })

    pairs = pd.DataFrame(rows).sort_values(
        "cosine_similarity", ascending=False
    )
    pairs.to_csv(OUTPUT_DIR / "word2vec_pairwise_similarity.csv", index=False)
    pairs.head(5).to_csv(OUTPUT_DIR / "word2vec_top5_pairs.csv", index=False)

    # Save nearest neighbours for each selected token.
    neighbors = []
    for token in available:
        for neighbor, score in model.wv.most_similar(token, topn=5):
            neighbors.append({
                "token": token,
                "neighbor": neighbor,
                "cosine_similarity": float(score),
            })
    pd.DataFrame(neighbors).to_csv(
        OUTPUT_DIR / "word2vec_nearest_neighbors.csv", index=False
    )

    print("\nTop 5 most similar selected-token pairs:")
    print(pairs.head(5).to_string(index=False))
    print("\nSaved Word2Vec model and results to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
