
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


---

## Exercise 2: Word2Vec

### Objective

Word2Vec learns numerical representations of tokens from their surrounding contexts. Unlike character-trigram embeddings, which capture spelling patterns, Word2Vec learns relationships from token sequences.

### Methodology

The experiment uses Gensim's Skip-gram Word2Vec implementation and the combined corpus prepared in Lab 2.

| Parameter | Value |
|---|---:|
| Corpus records | 73,918 |
| Vocabulary size | 113,215 |
| Embedding dimensions | 50 |
| Context window | 5 |
| Training epochs | 10 |
| Minimum token frequency | 1 |
| Random seed | 42 |

The same 20 selected tokens were used to make the comparison with the custom approaches more consistent. Word2Vec found 19 of the 20 tokens. The token `buil` was missing because the Word2Vec tokenizer and the BPE tokenizer used for token selection produce different vocabularies.

### Five most similar selected-token pairs

| Token 1 | Token 2 | Cosine similarity |
|---|---|---:|
| error | exception | 0.7509 |
| wait | attempt | 0.6756 |
| OPTIONS | verse | 0.6594 |
| script | attempt | 0.6532 |
| class | return | 0.6325 |

### Observations

The pair `error` and `exception` received the highest similarity. This is plausible because the two tokens are closely related in programming terminology and can occur in similar contexts.

The similarity between `wait` and `attempt` shows that tokens with different meanings can also become close when they occur in similar contexts.

Unlike character-trigram embeddings, Word2Vec learns from surrounding tokens. However, its results still depend on the training corpus, tokenization, and training parameters.

### Generated files

- `output/word2vec/word2vec.model`
- `output/word2vec/word2vec_embeddings.npy`
- `output/word2vec/word2vec_embedding_table.json`
- `output/word2vec/word2vec_nearest_neighbors.csv`
- `output/word2vec/word2vec_pairwise_similarity.csv`
- `output/word2vec/word2vec_top5_pairs.csv`

---

## Exercise 3: Code2Vec-Inspired AST Path Embeddings

### Objective

This experiment explores representing source code using structural information extracted from abstract syntax trees (ASTs), rather than relying only on token spelling or local token co-occurrence.

### Methodology

Python's built-in `ast` module was used to parse Python source files. The implementation extracted simplified paths between terminal nodes in the syntax trees. Each path context was represented using the endpoint labels and the intervening AST path.

The path-context features were aggregated by source file and reduced to a lower-dimensional representation using Truncated Singular Value Decomposition (SVD).

**Implementation scope:** This is a Code2Vec-inspired AST path-context baseline, not a reproduction of the original neural Code2Vec architecture. It does not train the original model's learned path, token, and attention representations.

### Results

| Metric | Result |
|---|---:|
| Python files processed | 1,500 |
| AST path contexts extracted | 262,515 |
| Context features | 91,281 |
| Embedding dimensions | 32 |
| SVD explained variance ratio | 40.25% |

The experiment compared source-file embeddings using cosine similarity. Several file pairs received a similarity of 1.0. This indicates that the simplified representations can become identical for files with similar extracted structures; it does not prove that the files implement the same functionality.

### Limitations

1. The experiment processes only a subset of the available Python source files.
2. The built-in AST parser supports Python, rather than every language in the repository.
3. The representation uses simplified AST paths rather than the original Code2Vec neural architecture.
4. SVD reduces the context features and may discard structural information.
5. Similar extracted features can produce identical or very similar vectors, even when the files have different purposes.

### Generated files

- `output/code2vec/code2vec_inspired_file_embeddings.npy`
- `output/code2vec/code2vec_inspired_embedding_table.json`
- `output/code2vec/code2vec_inspired_summary.csv`
- `output/code2vec/code2vec_inspired_top_file_pairs.csv`

---

## Comparison of Embedding Approaches

| Approach | Main information captured | Main limitation |
|---|---|---|
| Character trigrams | Character and spelling patterns | Does not capture meaning or context |
| PPMI + SVD | Local token co-occurrence | Similar contexts do not necessarily imply similar meanings |
| Word2Vec | Token relationships learned from surrounding words | Depends on corpus, tokenization, and training parameters |
| Code2Vec-inspired AST paths | Simplified source-code syntax structure | Not the original neural Code2Vec architecture |

### Overall observations

The experiments demonstrate that different representations capture different kinds of information.

Character trigrams are simple and interpretable but primarily measure surface similarity. PPMI + SVD represents local co-occurrence patterns, while Word2Vec learns dense token vectors from surrounding contexts. The AST path experiment incorporates structural information from Python syntax trees.

Cosine similarity is useful for comparing these representations, but a high score does not automatically prove semantic equivalence. Each method must be interpreted according to the information it captures and its limitations.
