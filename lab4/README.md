
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

## Approach 2: Context-Based Co-occurrence Embeddings

To be implemented.
