# Lab 3 — Tokenization

## Objective

This lab explores different tokenization strategies and their effect on vocabulary size, sequence length, and embedding matrix size.

The experiment uses the software text corpus generated in Lab 2 from five open-source repositories:

- FastAPI
- Flask
- Pytest
- Requests
- Scikit-learn

The corpus contains 73,918 text records consisting of source-code text and commit messages.

---

# Exercise 1 — Tokenizer Comparison

Three tokenization approaches were implemented:

1. Character-level tokenization
2. Word-level tokenization
3. Subword-level tokenization using BPE

The same corpus and embedding dimension of 100 were used for comparison.

| Tokenizer | Vocabulary Size | Average Sequence Length | Embedding Matrix Size |
|---|---:|---:|---:|
| Character | 430 | 429.60 | ~0.16 MB |
| Word | 105,533 | 92.16 | ~40.25 MB |
| Subword BPE | 10,000 | 92.66 | ~3.81 MB |

### Observation

Character tokenization produces a very small vocabulary but much longer sequences.

Word tokenization produces a very large vocabulary, which results in a much larger embedding matrix.

Subword BPE provides a balance between vocabulary size and sequence length. It reduces the embedding matrix size substantially compared with word-level tokenization while keeping the sequence length close to word-level tokenization.

---

# Exercise 2 — Random Embedding Analysis

The BPE tokenizer was selected for the remaining experiments.

The 50 most frequent subword tokens were identified from the corpus.

A reproducible random embedding matrix was initialized with:

- Embedding dimension: 100
- Number of tokens: 50
- Random seed: 42

Cosine similarity was calculated between the embedding vectors.

For ranking the most and least similar embeddings, each token was ranked according to its average cosine similarity with the other 49 tokens.

The complete results are stored in:

- `top50_embedding_similarity.csv`
- `top10_most_similar.csv`
- `top10_least_similar.csv`

---

# Exercise 3 — Selected Tokens and Nearest Neighbours

Five tokens were selected from the top 50 tokens:

- `.`
- `/`
- `` ` ``
- `None`
- `*`

For each selected token, the five nearest tokens in the random embedding space were calculated using cosine similarity.

The results are stored in:

`selected_token_nearest_neighbors.csv`

The selected tokens were chosen from different positions in the top-50 list. At least two tokens should be confirmed to be different from the tokens selected by the student next to me, as required by the lab instructions.

---

# Exercise 4 — Pairwise Similarity

All pairwise cosine similarities between the five selected tokens were calculated.

There are 10 unique pairs because:

C(5,2) = 10

The results are stored in:

`selected_token_pair_similarity.csv`

Because the embeddings were initialized randomly, the similarities do not represent learned semantic relationships.

---

# Exercise 5 — Naive Algorithm for Improving Similarity

## Manual Algorithm

The following naive algorithm was designed to improve the similarity of related tokens:

1. Start with the randomly initialized token embeddings.
2. Tokenize the software corpus using the trained BPE tokenizer.
3. Build a co-occurrence matrix for the top 50 tokens.
4. Use a small context window around each token.
5. For every token, calculate the average embedding of its frequently co-occurring tokens.
6. Move the token embedding slightly toward this average embedding.
7. Normalize the embeddings.
8. Repeat the process for several iterations.
9. Compare the cosine similarities before and after the update.

The implementation uses:

- Context window size: 2
- Number of iterations: 5
- Update factor (alpha): 0.2

The algorithm is intentionally simple and is not a full neural embedding-training method.

---

# Exercise 6 — Similarity Improvement Results

The experiment was repeated after applying the naive co-occurrence-based update.

| Measurement | Value |
|---|---:|
| Average similarity before | 0.001399 |
| Average similarity after | 0.167324 |
| Increase | 0.165926 |

The largest observed improvements included:

| Token A | Token B | Before | After | Change |
|---|---|---:|---:|---:|
| `"` | `:` | 0.041833 | 0.696603 | 0.654770 |
| `from` | `import` | -0.106132 | 0.524922 | 0.631054 |
| `",` | `:` | -0.032003 | 0.516899 | 0.548902 |
| `"` | `{` | 0.145368 | 0.691563 | 0.546195 |
| `.` | `np` | -0.162024 | 0.374631 | 0.536655 |

### Observation

The average cosine similarity increased from approximately 0.0014 to 0.1673.

This demonstrates that the naive co-occurrence-based algorithm successfully moved embeddings of frequently co-occurring tokens closer together.

For example, the similarity between `from` and `import` increased from -0.1061 to 0.5249.

However, increased similarity does not automatically mean that the embeddings have become semantically better. Since the algorithm pulls embeddings toward their co-occurring neighbours, excessive smoothing can also make unrelated tokens more similar.

---

# Dataset Analysis

## Dataset Description

The corpus was generated during Lab 2 from five open-source software repositories.

It contains:

- 73,918 total records
- 71,544 commit messages
- 2,374 source-code records
- 31,754,812 total characters
- Average text length: 429.6 characters
- Maximum text length: 394,863 characters

Repository distribution:

| Repository | Records |
|---|---:|
| Scikit-learn | 35,031 |
| Pytest | 18,040 |
| FastAPI | 8,677 |
| Requests | 6,530 |
| Flask | 5,640 |

The corpus contains both natural-language commit messages and programming-language source code.

---

# Dataset Problems / Limitations

### 1. Mixed text types

The dataset contains both source code and commit messages. These have very different linguistic structures.

### 2. Programming symbols

The corpus contains many symbols such as:

- `.`
- `,`
- `(`
- `)`
- `=`
- `:`
- `/`

These become highly frequent tokens.

### 3. Uneven repository distribution

Scikit-learn contributes substantially more records than Flask or Requests. Therefore, the corpus does not contain an equal amount of data from each repository.

### 4. Very long records

Some records are extremely large, with the maximum text length reaching 394,863 characters.

### 5. Random embeddings

The initial embeddings are randomly generated and therefore do not contain semantic information.

### 6. Naive similarity improvement

The co-occurrence-based algorithm is a simple heuristic. It can cause over-smoothing, where many tokens become increasingly similar.

---

# Possible Dataset Uses

The dataset can potentially be used for:

- Software repository mining
- Commit message analysis
- Code-text classification
- Software documentation analysis
- Source-code language modelling
- Developer activity analysis
- Commit message clustering
- Code search and retrieval
- Software engineering NLP experiments
- Tokenization and embedding experiments

---

# Tools and Libraries

- Python
- Pandas
- NumPy
- Hugging Face Tokenizers
- BPE tokenizer
- Git
- GitHub
- VS Code

## Hugging Face Tokenizers

The `tokenizers` library from Hugging Face was used to train and load the BPE subword tokenizer.

---

# Output Files

The following experiment outputs were generated:

```text
lab3/output/
├── tokenizer_statistics.csv
├── random_embedding_matrix.npy
├── top50_embedding_similarity.csv
├── top10_most_similar.csv
├── top10_least_similar.csv
├── selected_token_nearest_neighbors.csv
├── selected_token_pair_similarity.csv
├── improved_embedding_matrix.npy
├── similarity_before_after.csv
└── top_similarity_improvements.csv

Conclusion
The experiment demonstrated the differences between character, word, and subword tokenization.
Character tokenization produced a small vocabulary but long sequences, while word tokenization produced a very large vocabulary and embedding matrix.
BPE subword tokenization provided a practical balance between vocabulary size and sequence length.
The random embedding experiment demonstrated cosine-similarity analysis, while the final experiment showed that a simple co-occurrence-based update can increase similarity between frequently co-occurring tokens.
The results also demonstrate an important limitation of naive embedding improvement: increasing numerical similarity alone does not guarantee meaningful semantic representations.