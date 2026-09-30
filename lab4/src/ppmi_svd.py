import os
import numpy as np
import pandas as pd

from tokenizers import Tokenizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

CORPUS_PATH = "lab2/data/processed/text_corpus.csv"
TOKENIZER_PATH = "lab3/data/subword_bpe_tokenizer.json"
SELECTED_TOKENS_PATH = "lab4/data/selected_tokens.csv"

OUTPUT_DIR = "lab4/output/ppmi"
EMBEDDING_DIM = 10
WINDOW_SIZE = 2
RANDOM_STATE = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD SELECTED TOKENS
# ============================================================

selected_df = pd.read_csv(SELECTED_TOKENS_PATH)

selected_tokens = selected_df["token"].astype(str).tolist()

print("=" * 70)
print("PPMI + SVD CONTEXT EMBEDDING")
print("=" * 70)

print(f"\nNumber of selected tokens: {len(selected_tokens)}")
print("Selected tokens:")
print(selected_tokens)


# ============================================================
# LOAD BPE TOKENIZER
# ============================================================

print("\nLoading BPE tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

vocab = tokenizer.get_vocab()

print(f"BPE vocabulary size: {len(vocab)}")


# Create token -> ID mapping
token_to_id = vocab

# Context vocabulary consists of all BPE tokens
context_vocab_size = len(vocab)


# ============================================================
# MAP SELECTED TOKENS TO TOKEN IDs
# ============================================================

selected_token_ids = {}

for token in selected_tokens:
    if token in token_to_id:
        selected_token_ids[token] = token_to_id[token]
    else:
        print(f"WARNING: '{token}' not found in BPE vocabulary.")


print(
    f"\nSelected tokens found in BPE vocabulary: "
    f"{len(selected_token_ids)}/{len(selected_tokens)}"
)


# Reverse mapping for output
id_to_token = {idx: token for token, idx in token_to_id.items()}


# ============================================================
# BUILD CO-OCCURRENCE MATRIX
# ============================================================

print("\nBuilding co-occurrence matrix...")
print(f"Context window: +/- {WINDOW_SIZE}")

# Map selected token ID -> row number
target_id_to_row = {
    token_id: row
    for row, token_id in enumerate(selected_token_ids.values())
}

# We build the matrix using coordinate lists first.
rows = []
cols = []
data = []

corpus = pd.read_csv(CORPUS_PATH)

total_records = len(corpus)

print(f"Corpus records: {total_records:,}")

text_column = "text"

if text_column not in corpus.columns:
    raise ValueError(
        f"Expected column '{text_column}' not found. "
        f"Available columns: {list(corpus.columns)}"
    )


# ------------------------------------------------------------
# Process every document
# ------------------------------------------------------------

for doc_index, text in enumerate(corpus[text_column].fillna("")):

    if doc_index % 5000 == 0:
        print(f"Processed {doc_index:,}/{total_records:,} records...")

    # Convert document to BPE token IDs
    encoded = tokenizer.encode(str(text))

    token_ids = encoded.ids

    # Look for selected target tokens
    for position, token_id in enumerate(token_ids):

        if token_id not in target_id_to_row:
            continue

        target_row = target_id_to_row[token_id]

        # Determine context window
        start = max(0, position - WINDOW_SIZE)
        end = min(len(token_ids), position + WINDOW_SIZE + 1)

        for context_position in range(start, end):

            # Do not count the target token itself
            if context_position == position:
                continue

            context_id = token_ids[context_position]

            rows.append(target_row)
            cols.append(context_id)
            data.append(1)


# ============================================================
# CREATE SPARSE CO-OCCURRENCE MATRIX
# ============================================================

print("\nCreating sparse co-occurrence matrix...")

cooccurrence_matrix = csr_matrix(
    (data, (rows, cols)),
    shape=(len(selected_token_ids), context_vocab_size),
    dtype=np.float64
)

# Combine duplicate entries
cooccurrence_matrix.sum_duplicates()

print(
    "Co-occurrence matrix shape:",
    cooccurrence_matrix.shape
)

print(
    "Non-zero co-occurrence entries:",
    cooccurrence_matrix.nnz
)


# Save raw co-occurrence matrix
np.save(
    os.path.join(OUTPUT_DIR, "cooccurrence_matrix.npy"),
    cooccurrence_matrix.toarray()
)


# ============================================================
# CALCULATE PPMI
# ============================================================

print("\nCalculating PPMI...")

# Total number of co-occurrences
total_count = cooccurrence_matrix.sum()

# Sum over each target token
row_sums = np.asarray(
    cooccurrence_matrix.sum(axis=1)
).flatten()

# Sum over each context token
col_sums = np.asarray(
    cooccurrence_matrix.sum(axis=0)
).flatten()


# Convert sparse matrix to COO format
coo = cooccurrence_matrix.tocoo()

ppmi_data = []

for row, col, count in zip(coo.row, coo.col, coo.data):

    # P(target, context)
    p_joint = count / total_count

    # P(target)
    p_target = row_sums[row] / total_count

    # P(context)
    p_context = col_sums[col] / total_count

    # PMI
    pmi = np.log2(
        p_joint / (p_target * p_context)
    )

    # Positive PMI
    ppmi = max(pmi, 0.0)

    ppmi_data.append(ppmi)


ppmi_matrix = csr_matrix(
    (ppmi_data, (coo.row, coo.col)),
    shape=cooccurrence_matrix.shape
)

ppmi_matrix.eliminate_zeros()

print("PPMI matrix shape:", ppmi_matrix.shape)
print("Non-zero PPMI entries:", ppmi_matrix.nnz)


# Save PPMI matrix
np.save(
    os.path.join(OUTPUT_DIR, "ppmi_matrix.npy"),
    ppmi_matrix.toarray()
)


# ============================================================
# APPLY SVD
# ============================================================

print("\nApplying SVD...")

# Number of components cannot exceed the useful matrix dimensions
max_components = min(
    ppmi_matrix.shape[0] - 1,
    ppmi_matrix.shape[1] - 1
)

n_components = min(
    EMBEDDING_DIM,
    max_components
)

svd = TruncatedSVD(
    n_components=n_components,
    random_state=RANDOM_STATE
)

embeddings = svd.fit_transform(ppmi_matrix)

print("Embedding matrix shape:", embeddings.shape)

print(
    "Explained variance ratio:",
    svd.explained_variance_ratio_.sum()
)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

embeddings_normalized = normalize(
    embeddings,
    norm="l2"
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "ppmi_svd_embeddings.npy"
    ),
    embeddings_normalized
)


# ============================================================
# CALCULATE PAIRWISE COSINE SIMILARITY
# ============================================================

print("\nCalculating pairwise cosine similarity...")

similarity_matrix = cosine_similarity(
    embeddings_normalized
)


# ============================================================
# CREATE PAIRWISE RESULTS
# ============================================================

token_list = list(selected_token_ids.keys())

pairwise_results = []

for i in range(len(token_list)):

    for j in range(i + 1, len(token_list)):

        pairwise_results.append(
            {
                "token_1": token_list[i],
                "token_2": token_list[j],
                "cosine_similarity": similarity_matrix[i, j]
            }
        )


pairwise_df = pd.DataFrame(pairwise_results)

pairwise_df = pairwise_df.sort_values(
    "cosine_similarity",
    ascending=False
)


# Save all pairwise similarities
pairwise_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ppmi_svd_pairwise_similarity.csv"
    ),
    index=False
)


# ============================================================
# TOP 5 MOST SIMILAR PAIRS
# ============================================================

top5 = pairwise_df.head(5)

top5.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ppmi_svd_top5_pairs.csv"
    ),
    index=False
)


# ============================================================
# SAVE EMBEDDING TABLE
# ============================================================

embedding_columns = [
    f"dimension_{i + 1}"
    for i in range(embeddings_normalized.shape[1])
]

embedding_df = pd.DataFrame(
    embeddings_normalized,
    columns=embedding_columns
)

embedding_df.insert(
    0,
    "token",
    token_list
)

embedding_df = embedding_df.merge(
    selected_df,
    on="token",
    how="left"
)

embedding_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ppmi_svd_embedding_table.csv"
    ),
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 MOST SIMILAR TOKEN PAIRS")
print("=" * 70)

print(
    top5.to_string(index=False)
)


print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"""
{OUTPUT_DIR}/cooccurrence_matrix.npy
{OUTPUT_DIR}/ppmi_matrix.npy
{OUTPUT_DIR}/ppmi_svd_embeddings.npy
{OUTPUT_DIR}/ppmi_svd_pairwise_similarity.csv
{OUTPUT_DIR}/ppmi_svd_top5_pairs.csv
{OUTPUT_DIR}/ppmi_svd_embedding_table.csv
"""
)

print("Experiment completed successfully.")