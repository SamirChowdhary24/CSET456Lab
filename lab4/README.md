
# Lab 4: Embeddings

## Exercise 1: Custom Embedding Approaches

This exercise explores two approaches to generating token embeddings and evaluates their results using pairwise cosine similarity.

The experiments reuse the BPE tokenizer developed in Lab 3 and the combined source-code and commit-message corpus prepared in Lab 2.

## Approach 1: Character Trigram Embeddings

### Idea and methodology

My first approach represents each token using overlapping sequences of three characters, called character trigrams.

I chose this approach because it is straightforward to implement and allows me to investigate how much similarity can be captured using the spelling of tokens alone, without training a neural network.

For example, the token `import` produces the trigrams `imp`, `mpo`, `por` and `ort`.

I constructed a vocabulary containing every unique trigram found in the selected tokens. Each token was then represented by a vector containing the frequency of each trigram.

The implementation converts tokens to lowercase before extracting trigrams.

### Token selection

I selected 20 tokens from the BPE tokenizer vocabulary: 10 manually and 10 randomly using a fixed random seed of 42.

**Manual tokens:** import, from, class, def, return, function, test, assert, error, exception

**Random tokens:** script, OPTIONS, Checking, wait, comm, buil, attempt, RidgeCV, verse, Model

The manual selection includes programming-related terms so I can examine their relationships. Random selection provides additional tokens that were not specifically chosen for their similarity.

The selected tokens are saved in `data/selected_tokens.csv`.

### Embedding and similarity statistics

| Metric | Result |
|---|---:|
| Selected tokens | 20 |
| Unique character trigrams | 67 |
| Embedding matrix dimensions | 20 × 67 |
| Unique token pairs | 190 |
| Similarity measure | Cosine similarity |

I calculated cosine similarity between every unique pair of token embeddings, excluding self-comparisons.

### Five most similar pairs

| Token 1 | Token 2 | Cosine similarity |
|---|---|---:|
| exception | OPTIONS | 0.507093 |
| function | OPTIONS | 0.365148 |
| function | exception | 0.308607 |
| class | assert | 0.288675 |
| import | def | 0.000000 |

The fifth pair has a similarity of zero and is tied with other pairs. Therefore, it should not be interpreted as uniquely the fifth most similar pair.

### Do these similarities make sense?

The results make sense when interpreted as **spelling similarity**, rather than semantic similarity.

`exception` and `OPTIONS` share several character trigrams after conversion to lowercase, resulting in the highest cosine similarity.

Similarly, `function` and `exception` share character sequences despite representing different programming concepts.

`class` and `assert` share the trigram `ass`, producing a positive similarity score.

However, the approach does not understand what the tokens mean or how they are used in source code.

### Representation failures

**Failure 1: Unrelated tokens can receive high similarity.**

`exception` and `OPTIONS` have the highest similarity score, even though they represent different concepts. Their similarity comes from shared spelling patterns rather than a meaningful relationship.

**Failure 2: Related programming tokens can receive zero similarity.**

`import` and `def` are both Python keywords, but their cosine similarity is zero because they share no character trigrams. The representation cannot recognize their relationship through their programming context.

**Additional limitation: Case sensitivity is lost.**

The implementation converts all tokens to lowercase. Consequently, it cannot distinguish tokens whose capitalization carries meaning, such as `Model` and `model`.

### Observations

Character trigram embeddings are simple, interpretable and capable of detecting overlapping spelling patterns. However, they cannot reliably represent semantic or contextual relationships.

This motivates the second approach, which will construct embeddings using token co-occurrence in the Lab 2 corpus.

### Generated files

- `data/selected_tokens.csv`
- `output/character_ngram_embeddings.npy`
- `output/character_ngram_vocabulary.csv`
- `output/character_ngram_pairwise_similarity.csv`
- `output/character_ngram_top5_pairs.csv`

## Approach 2: PPMI + SVD Context Embeddings

### Objective

The second embedding approach was designed to capture the contextual usage of tokens rather than their character-level form.

The main idea was based on the distributional hypothesis: tokens that occur in similar contexts may have similar representations.

The pipeline used was:

BPE Tokenization → Context Window → Co-occurrence Matrix → PPMI → SVD → Dense Embeddings → Cosine Similarity

### Token Selection

The same 20 tokens selected for Approach 1 were reused so that both embedding approaches could be compared fairly.

The selection consisted of 10 manually selected tokens and 10 randomly selected tokens.

### Context Representation

The BPE tokenizer created in Lab 3 was reused.

For every occurrence of one of the selected tokens, a context window of 2 tokens on either side was considered.

For example:

    token₋₂ token₋₁ TARGET token₊₁ token₊₂

The surrounding tokens were counted as contexts for the target token.

The context vocabulary consisted of the complete BPE vocabulary of 10,000 tokens.

This produced a:

- Co-occurrence matrix: 20 × 10,000
- Non-zero co-occurrence entries: 37,054

### PPMI Transformation

Raw co-occurrence counts were converted into Positive Pointwise Mutual Information (PPMI).

PPMI was used to give greater importance to context-token combinations that occurred more often than would be expected from their individual frequencies.

Negative PMI values were replaced with zero.

The resulting PPMI matrix contained:

- Matrix size: 20 × 10,000
- Non-zero PPMI entries: 23,099

### SVD Dimensionality Reduction

Truncated Singular Value Decomposition (SVD) was applied to the PPMI matrix to obtain dense lower-dimensional embeddings.

The final embedding dimension was 10.

The resulting embedding matrix had the shape:

    20 × 10

The selected SVD dimensions explained approximately 68.49% of the variance captured by the decomposition.

### Pairwise Cosine Similarity

Cosine similarity was calculated for every unique pair of the 20 token embeddings.

Number of unique pairs:

    20 × 19 / 2 = 190

The five most similar pairs were:

| Token 1 | Token 2 | Cosine Similarity |
|---|---|---:|
| Checking | RidgeCV | 0.9417 |
| wait | attempt | 0.9349 |
| OPTIONS | Checking | 0.8972 |
| Checking | comm | 0.8921 |
| OPTIONS | RidgeCV | 0.8672 |

### Interpretation

The similarities should be interpreted as contextual similarities rather than direct semantic similarity or synonymy.

For example, `wait` and `attempt` received a high similarity of 0.9349. Although their meanings are different, they occurred in sufficiently similar contexts in the corpus for their PPMI-based representations to become close.

The pair `return` and `assert` also received a relatively high similarity of 0.8240. This is reasonably plausible because both tokens commonly occur in Python programming contexts.

Therefore, the method captures patterns of token usage in the corpus, but these patterns do not always correspond directly to semantic relationships.

### Representation Failures

#### Failure 1: False Contextual Similarity

The tokens `wait` and `attempt` obtained a cosine similarity of 0.9349 despite having different meanings.

This demonstrates that PPMI + SVD can produce high similarity between unrelated tokens when they occur in similar local contexts.

#### Failure 2: Missed Conceptual Similarity

The tokens `class` and `def` obtained a cosine similarity of approximately 0.0485.

Both are important Python keywords used for defining program structures, but their local contexts can be substantially different.

This demonstrates that the representation may fail to capture higher-level conceptual relationships when related tokens are used in different contextual patterns.

### Limitations

The PPMI + SVD approach has several limitations:

1. It depends strongly on the contexts present in the training corpus.
2. A small context window may miss longer-range relationships between tokens.
3. Similar contexts do not necessarily imply similar meanings.
4. SVD reduces the high-dimensional PPMI matrix, which may discard some information.
5. The resulting vectors can contain negative values, so negative cosine similarity does not mean that two tokens have opposite meanings.

### Output Files

The experiment generated the following files:

- `lab4/output/ppmi/cooccurrence_matrix.npy`
- `lab4/output/ppmi/ppmi_matrix.npy`
- `lab4/output/ppmi/ppmi_svd_embeddings.npy`
- `lab4/output/ppmi/ppmi_svd_pairwise_similarity.csv`
- `lab4/output/ppmi/ppmi_svd_top5_pairs.csv`
- `lab4/output/ppmi/ppmi_svd_embedding_table.csv`
