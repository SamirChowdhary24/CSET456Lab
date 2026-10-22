
from pathlib import Path
import ast
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

LAB4_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = LAB4_DIR.parent
DATA_PATH = PROJECT_DIR / "lab2" / "data" / "processed" / "clean_source_code_dataset.csv"
OUTPUT_DIR = LAB4_DIR / "output" / "code2vec"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILES = 1500
MAX_PATH_LENGTH = 8
MAX_CONTEXTS_PER_FILE = 200
VECTOR_SIZE = 32
SEED = 42


def terminal_nodes(node, path=()):
    """Return (terminal label, AST path) pairs for a syntax tree."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        label = getattr(node, "id", None) or getattr(node, "name", None)
        if label is None and isinstance(node, ast.Constant):
            label = str(node.value)
        if label is None:
            label = type(node).__name__
        return [(str(label), path + (type(node).__name__,))]

    result = []
    for child in children:
        result.extend(terminal_nodes(child, path + (type(node).__name__,)))
    return result


def extract_contexts(source):
    """Extract simplified AST path contexts between terminal nodes."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return []

    leaves = terminal_nodes(tree)
    contexts = []
    for i in range(len(leaves)):
        label_a, path_a = leaves[i]
        for j in range(i + 1, min(len(leaves), i + 16)):
            label_b, path_b = leaves[j]
            common = 0
            while (
                common < min(len(path_a), len(path_b))
                and path_a[common] == path_b[common]
            ):
                common += 1
            path_length = len(path_a) + len(path_b) - 2 * common
            if path_length <= MAX_PATH_LENGTH:
                path = ">".join(path_a[common:] + path_b[common:][::-1])
                contexts.append((label_a, path, label_b))
            if len(contexts) >= MAX_CONTEXTS_PER_FILE:
                return contexts
    return contexts


def main():
    print("Loading cleaned source-code dataset...")
    df = pd.read_csv(DATA_PATH, usecols=["file_path", "file_extension", "source_code"])
    df = df[df["file_extension"].isin([".py", ".pyi"])].dropna(subset=["source_code"])
    df = df.head(MAX_FILES)
    print(f"Python source files selected: {len(df):,}")

    documents = []
    total_contexts = 0
    for source in df["source_code"].astype(str):
        contexts = extract_contexts(source)
        total_contexts += len(contexts)
        counts = Counter()
        for left, path, right in contexts:
            counts[f"LEFT={left}|PATH={path}|RIGHT={right}"] += 1
        documents.append(counts)

    if not any(documents):
        raise RuntimeError("No AST path contexts were extracted.")

    print(f"AST path contexts extracted: {total_contexts:,}")
    vectorizer = DictVectorizer()
    matrix = vectorizer.fit_transform(documents)
    print(f"Context matrix shape: {matrix.shape}")

    dimension = min(VECTOR_SIZE, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if dimension < 1:
        raise RuntimeError("Not enough code documents or contexts for SVD.")

    svd = TruncatedSVD(n_components=dimension, random_state=SEED)
    embeddings = normalize(svd.fit_transform(matrix))
    np.save(OUTPUT_DIR / "code2vec_inspired_file_embeddings.npy", embeddings)

    pd.DataFrame({
        "file_path": df["file_path"].tolist(),
        "embedding": [row.tolist() for row in embeddings],
    }).to_json(OUTPUT_DIR / "code2vec_inspired_embedding_table.json",
               orient="records", indent=2)

    # Save the most similar pairs of source files.
    similarity = cosine_similarity(embeddings)
    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            pairs.append({
                "file_1": df.iloc[i]["file_path"],
                "file_2": df.iloc[j]["file_path"],
                "cosine_similarity": float(similarity[i, j]),
            })
    pairs_df = pd.DataFrame(pairs).sort_values(
        "cosine_similarity", ascending=False
    )
    pairs_df.head(100).to_csv(
        OUTPUT_DIR / "code2vec_inspired_top_file_pairs.csv", index=False
    )

    # Summary metrics for documentation.
    summary = pd.DataFrame([{
        "files_processed": len(df),
        "ast_contexts_extracted": total_contexts,
        "context_features": matrix.shape[1],
        "embedding_dimensions": dimension,
        "svd_explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
        "method": "AST path-context features + TF-IDF-like document weighting via SVD",
    }])
    summary.to_csv(OUTPUT_DIR / "code2vec_inspired_summary.csv", index=False)

    print("\nMost similar source-file pairs:")
    print(pairs_df.head(5).to_string(index=False))
    print("\nSaved Code2Vec-inspired outputs to:", OUTPUT_DIR)
    print("Note: this is not the original neural Code2Vec architecture.")


if __name__ == "__main__":
    main()
