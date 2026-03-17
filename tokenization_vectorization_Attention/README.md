# Week 2:  Tokenization, Vectorization & Attention



---

##  Table of Contents

| # | Topic | 

|---|-------|

| 1 | [Tokenization](#1-tokenization) 

| 1.1 | [Why We Need Tokenization](#11-why-we-need-tokenization) 

| 1.2 | [Tokenization Strategies](#12-tokenization-strategies) 

| 1.3 | [Byte Pair Encoding (BPE) — Step by Step](#13-byte-pair-encoding-bpe--step-by-step) 

| 1.4 | [WordPiece and SentencePiece](#15-wordpiece-and-sentencepiece)  

| 1.5 | [Vocabulary Size Tradeoffs](#16-vocabulary-size-tradeoffs) 

| 1.6 | [Special Tokens](#17-special-tokens) 


| 2 | [Vectorization / Embeddings](#2-vectorization--embeddings) 

| 2.1 | [Why Vectors?](#21-why-vectors)  

| 2.2 | [Evolution of Word Representations](#22-evolution-of-word-representations)  

| 2.3 | [How Embedding Layers Work](#23-how-embedding-layers-work)  

| 3 | [Positional Encodings](#3-positional-encodings) 

| 3.1 | [Why Position Matters](#31-why-position-matters) 

| 3.2 | [Sinusoidal Positional Encodings](#32-sinusoidal-positional-encodings)  

| 3.3 | [Modern Positional Encodings](#33-modern-positional-encodings) 

| 4 | [The Attention Mechanism](#4-the-attention-mechanism) 

| 4.1 | [Intuition](#41-intuition) 

| 4.2 | [The Attention Formula](#42-the-attention-formula)  

| 4.3 | [Why Scale by √dk?](#44-why-scale-by-dk) 
 

| — | [Week 2 Summary](#week-2-summary)  

| — | [Exercises](#exercises) 

| — | [Further Reading](#further-reading) 

---

## 1. Tokenization

### 1.1 Why We Need Tokenization

Computers don't understand text. They understand numbers. **Tokenization** is the process of converting raw text into a sequence of **tokens** — discrete units each mapped to an integer ID. This is **always the first step** in any NLP pipeline.

```
Raw Text         →   Tokenizer   →   Token IDs   →   Neural Network
"Hello, world!"                     [9906, 11, 1917, 0]
```

```python
# The fundamental problem:
text = "Hello, world!"
# Computer sees: just a sequence of bytes/characters
# We need to convert this into numbers that a neural network can process

# After tokenization (using GPT-4's tokenizer):
# "Hello, world!" -> [9906, 11, 1917, 0]
# Each number is an index into a vocabulary of ~100,000 tokens
```

#### Why not just use raw byte values?

While technically possible, raw bytes (0–255) create extremely long sequences, and the model must learn to compose characters into meaningful words from scratch — an enormous challenge. Tokenization pre-packages text into meaningful units, giving the model a head start.

---

### 1.2 Tokenization Strategies

There are four main strategies, each with different tradeoffs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│        TOKENIZATION STRATEGY COMPARISON                                 │
│        Input: "unhappiness is contagious"                               │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│   WORD       │   SUBWORD    │  CHARACTER   │    BYTE      │             │
│              │   (BPE)      │              │              │             │
├──────────────┼──────────────┼──────────────┼──────────────┤             │
│ unhappiness  │ un           │ u            │ 75           │             │
│ is           │ happi        │ n            │ 6E           │             │
│ contagious   │ ness         │ h            │ 68           │             │
│              │ is           │ a            │ 61           │             │
│              │ cont         │ p            │ 70           │             │
│              │ agious       │ p            │ 70           │             │
│              │              │ i            │ 69           │             │
│              │              │ ...          │ ...          │             │
├──────────────┼──────────────┼──────────────┼──────────────┤             │
│  3 tokens    │  6 tokens    │  25 tokens   │  25 bytes    │             │
│  Vocab:170K+ │  Vocab:32-   │  Vocab:~256  │  Vocab:256   │             │
│              │  128K        │              │              │             │
├──────────────┴──────────────┴──────────────┴──────────────┤             │
│  ← More tokens, smaller vocab    Fewer tokens, larger vocab →           │
│                                                                         │
│  ★ Modern LLMs use SUBWORD (BPE) = best balance between characters and full words.                      │
│    Fewer tokens = faster inference                                      │
│    Smaller vocab = fewer parameters                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Character-Level Tokenization

Split text into individual characters.

```python
# Character-level tokenization
text = "Hello world"
tokens = list(text)
print(tokens)  # ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

# Vocabulary is tiny: ~256 characters (ASCII) or ~150,000 (Unicode)
# But sequences become VERY long: a 1000-word article = ~5000 characters
# This makes it hard for the model to learn long-range dependencies

# Pros: Small vocabulary, handles ANY text (no unknown tokens)
# Cons: Very long sequences, hard to capture word-level meaning
```

#### Word-Level Tokenization

Split text into words (by whitespace and punctuation).

```python
# Word-level tokenization
text = "The cat sat on the mat."
tokens = text.split()
print(tokens)  # ['The', 'cat', 'sat', 'on', 'the', 'mat.']

# Problems:
# 1. Huge vocabulary: English has ~170,000 words + names + technical terms
# 2. Can't handle new words: "ChatGPT" would be [UNK] if not in vocabulary
# 3. Morphology is lost: "run", "running", "runner" are 3 separate tokens
#    with no shared representation
# 4. Different languages: Chinese/Japanese don't use spaces between words

# Pros: Intuitive, short sequences
# Cons: Huge vocab, can't handle unseen words, no morphological sharing
```

#### Subword Tokenization 

Modern LLMs use subword tokenization — a middle ground. Common words stay as single tokens; rare words are broken into meaningful subword units.

```python
# Subword tokenization (what modern LLMs actually use)
# "unhappiness" -> ["un", "happiness"]  or  ["un", "happi", "ness"]
# "ChatGPT"    -> ["Chat", "G", "PT"]
# "the"        -> ["the"]  (common word stays whole)

# Benefits:
#  Manageable vocabulary size (32K–128K tokens)
#  Handles ANY word (even new ones like "ChatGPT")
#  Shares representations: "unhappy" and "unhelpful" share "un"
#  Good sequence lengths (not too long, not too short)
```

#### The Tokenization Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                  TOKENIZATION PIPELINE                       │
│                                                              │
│  Raw Text          Tokenizer         Token IDs              │
│  ──────────        ─────────         ─────────              │
│  "Hello world"  →  BPE/WordPiece  →  [4521, 328, 1917]     │
│                                           │                  │
│                                    Embedding Layer           │
│                                    (Lookup Table)            │
│                                           │                  │
│                                    Dense Vectors             │
│                                    (768-dim each)            │
│                                           │                  │
│                                    Transformer Model         │
└──────────────────────────────────────────────────────────────┘
```

---

### 1.3 Byte Pair Encoding (BPE) — Step by Step

BPE is the most popular subword tokenization algorithm, used by **GPT-2, GPT-3, GPT-4, Llama**, and most modern LLMs. The algorithm is surprisingly simple:

#### The BPE Algorithm

```
1. Start with a vocabulary of individual characters (bytes)
2. Count all adjacent pairs of tokens in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 until the desired vocabulary size is reached
```

#### BPE Merge Steps — Visual Walkthrough

```
Training corpus: "low low low low low lowest lowest newer newer newer wider wider wider"

┌─────────────────────────────────────────────────────────────────┐
│  STEP 0: Initial — individual characters                        │
│  "lower" = [l, o, w, e, r]  →  5 tokens                        │
│                                                                 │
│  Count all adjacent pairs across entire corpus:                 │
│    ('l','o'): 7   ('o','w'): 7   ('e','r'): 6   ...           │
└─────────────────────────────────────────────────────────────────┘
         │  Most frequent pair: ('l','o') or ('o','w') → tie → pick first
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Merge l + o → lo                                       │
│  "lower" = [lo, w, e, r]   →  4 tokens                         │
│                                                                 │
│  New pair counts:                                               │
│    ('lo','w'): 7   ('e','r'): 6   ...                          │
└─────────────────────────────────────────────────────────────────┘
         │  Most frequent pair: ('lo','w')
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Merge lo + w → low                                     │
│  "lower" = [low, e, r]   →  3 tokens                           │
│                                                                 │
│  New pair counts:                                               │
│    ('e','r'): 6   ('low', ''): 5  ...                          │
└─────────────────────────────────────────────────────────────────┘
         │  Most frequent pair: ('e','r')
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Merge e + r → er                                       │
│  "lower" = [low, er]   →  2 tokens                             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Merge low + er → lower                                 │
│  "lower" = [lower]   →  1 token                            │
│                                                                 │
│  But: "lowest" = [low, est]   ← stays as 2 tokens              │
│       "newer"  = [newer]      ← becomes 1 token                │
│       "wider"  = [wider]      ← becomes 1 token                │
└─────────────────────────────────────────────────────────────────┘

Key insight: Common words become single tokens.
             Rare words remain as multiple subwords.
```

---



### 1.4 WordPiece and SentencePiece

#### WordPiece (Used in BERT)

Similar to BPE but uses a **different scoring criterion**. Instead of merging the most frequent pair, WordPiece merges the pair that **maximizes the likelihood of the training data**.

```
WordPiece notation:
  "unhappiness" → ["un", "##happi", "##ness"]
  "playing"     → ["play", "##ing"]
  "cat"         → ["cat"]   (in vocabulary, stays as one token)

The "##" prefix means "this token is a continuation of the previous word"
This helps the model know which tokens START new words.
```

```python
# BERT vocabulary example:
# Token ID 2003  -> "the"
# Token ID 7592  -> "hello"
# Token ID 2015  -> "##ing"   (subword continuation)
# Token ID 2094  -> "##ed"    (subword continuation)

# BERT tokenization of "Tokenization is fundamental to NLP.":
# ['[CLS]', 'token', '##ization', 'is', 'fundamental', 'to', 'nl', '##p', '.', '[SEP]']
# Note: BERT automatically adds [CLS] and [SEP] special tokens
```

#### SentencePiece (Used in Llama, T5)

A **language-independent** tokenizer that treats the input as a raw byte stream, making it work for any language without pre-tokenization rules.

```
SentencePiece key differences:
  1. Treats spaces as regular characters (represented as '▁')
  2. Language-independent: no need for language-specific pre-processing
  3. Can use BPE or Unigram model internally

Encoding examples:
  "Hello world"  → ["▁Hello", "▁world"]
  The ▁ represents a space BEFORE the token (word boundary)

Works identically for languages without spaces.
```

```python
import sentencepiece as spm

# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    input='training_data.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe'  # or 'unigram'
)

# Load and use
sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
tokens = sp.encode('Hello world', out_type=str)
ids    = sp.encode('Hello world', out_type=int)
```

---

### 1.5 Vocabulary Size Tradeoffs

| Model | Vocab Size | Tokenizer | Notes |
|-------|-----------|-----------|-------|
| GPT-2 | 50,257 | BPE | Byte-level BPE |
| GPT-4 / GPT-4o | 100,277 | BPE (cl100k_base) | Larger vocab for efficiency |
| BERT | 30,522 | WordPiece | Cased and uncased versions |
| Llama 2 | 32,000 | SentencePiece (BPE) | Relatively small vocab |
| Llama 3 / 3.1 | 128,256 | Tiktoken (BPE) | 4x larger than Llama 2 |
| DeepSeek V3 | 129,280 | BPE | Efficient multilingual coverage |

#### Tradeoff Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                  VOCABULARY SIZE TRADEOFFS                          │
│                                                                     │
│  SMALL VOCAB (32K)                 LARGE VOCAB (128K)               │
│  ─────────────────                 ──────────────────               │
│   Smaller embedding matrix        Shorter token sequences        │
│   Fewer model parameters          Faster inference               │
│   Longer token sequences          Larger embedding matrix        │
│   Slower inference                More parameters                │
│                                                                     │
│  Trend (2024–2026): Moving toward LARGER vocabularies (100K+)       │
│  Reason: Embedding matrix cost is SMALL relative to total model     │
│          size, and shorter sequences mean much faster inference.    │
│                                                                     │
│  Memory cost of embedding matrix:                                   │
│    GPT-2:   50K × 768  ×  2 bytes = ~77 MB                         │
│    Llama 3: 128K × 4096 × 2 bytes = ~1 GB  (tiny vs 8B total)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1.6 Special Tokens

Special tokens control how models process text — they're critical for understanding model behavior.

```python
special_tokens = {
    # ──────────────── BERT special tokens ────────────────
    "[CLS]":    "Classification token. Placed at START of input. "
                "Its final hidden state = the 'sentence embedding'.",

    "[SEP]":    "Separator token. Separates two sentences in pair tasks "
                "(e.g., 'Is sentence A entailed by sentence B?').",

    "[PAD]":    "Padding token. Makes all sequences same length in a batch. "
                "Attention mask ensures model IGNORES these.",

    "[MASK]":   "Mask token. Used in BERT's masked language modeling. "
                "Replaces a token the model must predict.",

    # ──────────────── GPT special tokens ─────────────────
    "<|endoftext|>": "End of text marker in GPT-2/3. Signals end of a "
                     "document during training.",

    "<|im_start|>":  "Start of a message in ChatGPT format.",
    "<|im_end|>":    "End of a message in ChatGPT format.",

    # ──────────────── Llama special tokens ───────────────
    "<s>":     "Beginning of sequence (BOS) in Llama.",
    "</s>":    "End of sequence (EOS) in Llama.",

    # ──────────────── Common across models ───────────────
    "<unk>":   "Unknown token. Fallback for characters not in vocabulary. "
               "Modern BPE tokenizers rarely produce this.",
}
```

#### Llama 3 Chat Template Example

```
<|begin_of_text|>          → Start of conversation

[System]
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.

[User]
<|start_header_id|>user<|end_header_id|>
What is the capital of France?

[Assistant]
<|start_header_id|>assistant<|end_header_id|>
The capital of France is Paris.

<|eot_id|>                 → End of this conversation block
```

> **Key Insight**: The special tokens define the *grammar* of how you talk to a model. Using the wrong template (e.g., using Llama 2 format with a Llama 3 model) will significantly degrade performance even if the model understands the underlying language perfectly.

---



## 2. Vectorization / Embeddings

### 2.1 Why Vectors?

Token IDs are arbitrary numbers — `token 5000` is not "more" than `token 3000` in any meaningful way. We need a representation where **similar meanings are close together** in a continuous space.

#### The Core Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│            TOKEN IDs vs EMBEDDINGS                                  │
├───────────────────────────┬─────────────────────────────────────────┤
│  RAW TOKEN IDs (Bad)      │  DENSE EMBEDDINGS (Good)                │
│                           │                                         │
│  "banana"  → ID: 2345     │  "banana"  → [0.1, 0.8, 0.1, ...]     │
│  "king"    → ID: 5123     │  "king"    → [0.8, 0.2, 0.9, ...]     │
│  "queen"   → ID: 8901     │  "queen"   → [0.7, 0.3, 0.9, ...]     │
│                           │                                         │
│  |5123-2345| = 2778       │  cos(king, queen)  = 0.95  ← CLOSE   │
│  |8901-5123| = 3778       │  cos(king, banana) = 0.05  ← FAR    │
│                           │                                         │
│  "banana" is CLOSER to    │  Semantic similarity = vector proximity │
│  "king" than "queen"!    │                                       │
│  This is meaningless.     │  The space captures meaning.            │
└───────────────────────────┴─────────────────────────────────────────┘
```

#### The Famous Word2Vec Result

```
king  -  man  +  woman  ≈  queen
─────────────────────────────────
This arithmetic WORKS in embedding space!
The embedding space encodes semantic relationships as geometric directions.
```

---

### 2.2 Evolution of Word Representations

#### Stage 1: One-Hot Encoding

```
Vocabulary: [the, cat, sat, on, mat]    (vocab_size = 5)

"cat" →  [0, 1, 0, 0, 0]
"mat" →  [0, 0, 0, 0, 1]

cosine_similarity("cat", "mat") = 0.0  ← No semantic information!
All words are equally distant from each other.
```

```python
import numpy as np

vocab_size = 5  # Vocabulary: ["the", "cat", "sat", "on", "mat"]

one_hot = {
    "the": np.array([1, 0, 0, 0, 0]),
    "cat": np.array([0, 1, 0, 0, 0]),
    "sat": np.array([0, 0, 1, 0, 0]),
    "on":  np.array([0, 0, 0, 1, 0]),
    "mat": np.array([0, 0, 0, 0, 1]),
}

# Problems:
# 1. HUGE vectors for real vocabularies (50,000-dimensional!)
# 2. ALL words are equally distant from each other
cosine_sim = np.dot(one_hot["cat"], one_hot["mat"])
print(f"Similarity(cat, mat) = {cosine_sim}")  # 0.0
# "cat" and "mat" have zero similarity — same as "cat" and "quantum_physics"
```

#### Stage 2: Word2Vec (2013)

Word2Vec learns dense vector representations where **semantically similar words are close together**. Two architectures:

```
┌─────────────────────────────────────────────────────────────────────┐
│              WORD2VEC: SKIP-GRAM ARCHITECTURE                       │
│                                                                     │
│  Training example from: "The cat sat on the mat"                   │
│                                                                     │
│    INPUT          HIDDEN LAYER        PREDICT CONTEXT               │
│  ─────────        ─────────────       ──────────────────            │
│  "cat"            Embedding!          "The"   p=0.12                │
│  One-Hot    →     [.3, .7, .1, .9]  → "sat"   p=0.15               │
│  [0,1,0,0,0]      300 dims           "on"    p=0.10                │
│  50K dims         W = learned        "the"   p=0.11                │
│                   weights            "mat"   p=0.09                │
│                                      ...                            │
│                                                                     │
│  KEY INSIGHT: After training, the HIDDEN LAYER WEIGHTS              │
│               ARE the word embeddings!                              │
│                                                                     │
│  Words in similar contexts (cat/dog) get similar hidden             │
│  representations → similar embeddings                               │
│                                                                     │
│  king - man + woman = queen  ← Vector arithmetic works!            │
└─────────────────────────────────────────────────────────────────────┘
```

**CBOW vs Skip-gram:**

| | CBOW | Skip-gram |
|---|------|-----------|
| Task | Predict center from context | Predict context from center |
| Speed | Faster to train | Slower to train |
| Rare words | Weaker | Stronger |
| Use case | Frequent words | Diverse corpus |

#### Stage 3: GloVe (Global Vectors)

Combines local context (Word2Vec) with global corpus statistics.

```
Key idea: The RATIO of co-occurrence probabilities encodes meaning:

  P(ice | solid)  / P(ice | gas)   = large  →  ice is solid
  P(steam | solid)/ P(steam | gas) = small  →  steam is gas
  P(water | solid)/ P(water | gas) ≈ 1      →  water is both

GloVe trains embeddings so: w_i · w_j + b_i + b_j ≈ log(X_ij)
where X_ij is the co-occurrence count of words i and j.
```

#### Stage 4: Contextual Embeddings (BERT, GPT) 

The **revolutionary** change: the same word gets *different* embeddings depending on context.

```
"bank" in different contexts:
  "I went to the bank to deposit money"  → bank = [0.8, 0.1, 0.9, ...]
  "We sat on the river bank"             → bank = [0.2, 0.7, 0.3, ...]

Word2Vec: "bank" = [0.3, 0.5, 0.2]   (always the SAME vector)
BERT:     "bank" = different vector based on ENTIRE surrounding context

This is why transformer-based models are so much better:
meaning shifts with context, just like in human language!
```

---

### 2.3 How Embedding Layers Work

An embedding layer is simply a **matrix lookup table**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                  EMBEDDING LAYER = LOOKUP TABLE                      │
│                                                                      │
│  Token IDs                  Embedding Matrix (50,257 × 768)          │
│  ──────────                 ─────────────────────────────────        │
│  9906 "Hello"               ID   dim0  dim1  dim2  ... dim767        │
│  11   ","          →        0   -.02   .15  -.08  ...   .03         │
│  1917 "world"               ...                                      │
│                             11   .41  -.12   .67  ...  -.23         │
│                             ...                                      │
│                             1917  .08   .55  -.31  ...   .19        │
│                             ...                                      │
│                             9906  .72  -.34   .15  ...   .48        │
│                                                                      │
│  Output:                                                             │
│  "Hello" → [.72, -.34, .15, ..., .48]   (768 dimensions)           │
│  ","     → [.41, -.12, .67, ..., -.23]                              │
│  "world" → [.08,  .55, -.31, ..., .19]                              │
│                                                                      │
│  Operation: embeddings[token_id]   ← just array indexing!           │
│                                                                      │
│  Parameters:  GPT-2: 50,257 × 768 = 38.6M                           │
│               Llama 3.1 8B: 128,256 × 4,096 = 525M                 │
│               Values are LEARNED during training via backprop        │
└──────────────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

class EmbeddingLayer:
    """
    An embedding layer is simply a lookup table.
    It maps each token ID to a dense vector of size `embedding_dim`.
    These vectors are LEARNED during training (they are parameters).
    """

    def __init__(self, vocab_size, embedding_dim):
        # Initialize with random values (will be learned during training)
        # Shape: (vocab_size, embedding_dim)
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        print(f"Embedding layer: {vocab_size} tokens x {embedding_dim} dimensions")
        print(f"Total parameters: {vocab_size * embedding_dim:,}")

    def forward(self, token_ids):
        """
        Look up embeddings for the given token IDs.
        This is just an array index operation — very fast!
        """
        return self.embeddings[token_ids]


# Example: GPT-2 scale embedding
embed = EmbeddingLayer(vocab_size=50257, embedding_dim=768)
# Embedding layer: 50257 tokens x 768 dimensions
# Total parameters: 38,597,376 (~38.6M parameters just for embeddings!)

# Look up embeddings for a token sequence
token_ids = [15496, 11, 995, 0]  # "Hello, world!"
embeddings = embed.forward(token_ids)

print(f"\nInput token IDs: {token_ids}")
print(f"Output shape: {embeddings.shape}")  # (4, 768)

# What does 768 dimensions mean?
# Each dimension captures some aspect of the token's meaning:
#   - Part of speech (noun, verb, adjective)
#   - Semantic category (animal, food, color)
#   - Syntactic role (subject, object)
#   - Sentiment (positive, negative)
#   - And many more subtle features

# Common embedding dimensions:
# BERT-base:     768     GPT-2:         768
# BERT-large:   1024     GPT-3:       12288
# Llama 3.1 8B: 4096     Llama 3.1 70B: 8192
```

---



## 3. Positional Encodings

### 3.1 Why Position Matters

Unlike RNNs that process tokens **sequentially** (inherently knowing position), transformers process **all tokens in parallel**. Without positional information, the model sees the input as a **bag of words** — order doesn't matter.

```
Without positional encoding:
  "dog bites man"  →  {dog, bites, man}  ← treated as a SET
  "man bites dog"  →  {man, bites, dog}  ← IDENTICAL to above!

With positional encoding:
  "dog"  at position 0  ←  different vector
  "dog"  at position 2  ←  different vector
  Now order matters!
```

#### Positional Encoding Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│               HOW POSITIONAL ENCODINGS ARE ADDED                    │
│                                                                     │
│  Token Embedding               Positional Encoding                  │
│  ────────────────              ────────────────────                  │
│  "The"  [.3, .7, .1, .9]  +   pos=0  [0.0, 1.0, 0.0, 1.0]       │
│  "cat"  [.5, .2, .8, .4]  +   pos=1  [0.8, 0.5, 0.01, 0.5]      │
│  "sat"  [.1, .6, .3, .7]  +   pos=2  [0.9, 0.1, 0.02, 0.1]      │
│         ↓                             ↓                             │
│         Element-wise ADDITION (same shape, just add!)               │
│         ↓                                                           │
│  "The"@pos0  [.3, 1.7, .1, 1.9]                                    │
│  "cat"@pos1  [1.3, .7, .81, .9]                                    │
│  "sat"@pos2  [1.0, .7, .32, .8]                                    │
│                                                                     │
│  The transformer now knows WHAT each token is AND WHERE it is!     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Sinusoidal Positional Encodings

The original Transformer paper ("Attention Is All You Need") used **sinusoidal functions** to generate positional encodings.

#### The Formula

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
  pos = position in the sequence (0, 1, 2, ...)
  i   = dimension index
  d_model = embedding dimension

Even dimensions → sine    Odd dimensions → cosine
```

#### Visual Intuition

```
┌─────────────────────────────────────────────────────────────────────┐
│              SINUSOIDAL PATTERN ACROSS POSITIONS                    │
│                                                                     │
│  Value                                                              │
│   1 ┤                                                               │
│     │  dim 0 (fast oscillation)                │
│   0 ┤──────────────────────────────────────── Position             │
│     │  dim 2 (medium)                                 │
│  -1 ┤  dim 6 (slow)                                             │
│     │  dim 7 (slow)                                             │
│     └────────────────────────────────────────                       │
│      pos0  pos5  pos10  pos15  pos20  pos25  pos30                  │
│                                                                     │
│  Low dimensions oscillate FAST  → fine-grained position info       │
│  High dimensions oscillate SLOW → coarse position info             │
│  Each position has a UNIQUE "fingerprint"                           │
│  Nearby positions have SIMILAR encodings (smooth interpolation)    │
└─────────────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Why sinusoidal?
    1. Unique encoding for each position
    2. Bounded values (always between -1 and 1)
    3. Can extrapolate to longer sequences than seen during training
    4. Relative positions can be computed as linear transformations:
       PE(pos+k) can be represented as a linear function of PE(pos)
    """
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len).reshape(-1, 1)  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)  # Even dimensions: sine
    PE[:, 1::2] = np.cos(position * div_term)  # Odd dimensions: cosine

    return PE

# Generate for a small example
PE = sinusoidal_positional_encoding(max_len=10, d_model=8)

print("Positional Encoding Matrix (10 positions x 8 dimensions):")
print(f"{'Pos':>4}", end="")
for d in range(8):
    print(f"  dim{d:1d}", end="")
print()

for pos in range(10):
    print(f"{pos:4d}", end="")
    for d in range(8):
        print(f" {PE[pos, d]:6.3f}", end="")
    print()

# How it's used in the transformer:
# final_embedding = token_embedding + positional_encoding
# This is simply element-wise addition!
```

---

### 3.3 Modern Positional Encodings

#### Rotary Position Embeddings (RoPE) 

Used by **Llama, Mistral, GPT-NeoX**, and most modern LLMs. Instead of adding positional information, RoPE **rotates** the query and key vectors.

```
Standard PE:  embedding = token_embed + pos_embed   (addition)
RoPE:         embedding = rotate(token_embed, position)

The rotation angle depends on the position.
Each pair of dimensions is rotated by a different angle.
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                HOW RoPE ROTATION WORKS                              │
│                                                                     │
│  For each pair of dimensions (dim 2i, dim 2i+1):                   │
│                                                                     │
│  θ = position / (10000^(2i/d_model))                               │
│                                                                     │
│  [x_new_2i    ]   [cos θ  -sin θ] [x_2i    ]                      │
│  [x_new_2i+1  ] = [sin θ   cos θ] [x_2i+1  ]                      │
│                                                                     │
│  This is a 2D rotation in the plane of each dimension pair.        │
│                                                                     │
│  Key Property: dot product of rotated vectors depends ONLY on      │
│  RELATIVE position, not absolute positions.                         │
│  q_pos3 · k_pos5 depends on distance (5-3=2), not on 3 or 5.      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def apply_rope(x, position, d_model):
    """
    Apply Rotary Position Embeddings to a vector.

    Args:
        x:        input vector of shape (d_model,)
        position: position index (integer)
        d_model:  dimension of the vector
    """
    output = np.zeros_like(x)

    for i in range(d_model // 2):
        # Rotation angle depends on position and dimension
        theta = position / (10000 ** (2 * i / d_model))

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Apply 2D rotation to each pair of dimensions
        # Rotation matrix: [[cos, -sin], [sin, cos]]
        output[2*i]     = x[2*i] * cos_theta - x[2*i+1] * sin_theta
        output[2*i + 1] = x[2*i] * sin_theta + x[2*i+1] * cos_theta

    return output

d_model = 8
x = np.random.randn(d_model)

print("RoPE Demonstration:")
print(f"Original vector: {x}")
print(f"Position 0: {apply_rope(x, 0, d_model)}")  # No rotation at position 0
print(f"Position 1: {apply_rope(x, 1, d_model)}")  # Slight rotation
print(f"Position 5: {apply_rope(x, 5, d_model)}")  # More rotation
```

#### ALiBi (Attention with Linear Biases)

```
Standard attention:  score(i,j) = Q_i · K_j / sqrt(d_k)
ALiBi attention:     score(i,j) = Q_i · K_j / sqrt(d_k) - m * |i - j|

Where:
  m   = head-specific slope (different for each attention head)
  |i-j| = absolute distance between token positions

Benefits:
   Simpler than RoPE (just a bias term)
   Good length extrapolation
   Used in BLOOM

Status: RoPE has become more dominant (2024-2026) for:
  - Better performance on long-context tasks
  - Works well with NTK-aware scaling for context extension
  - Adopted by Llama, Mistral, and most popular open-source models
```

#### Comparison of Positional Encoding Methods

| Method | Used By | Mechanism | Length Extrapolation |
|--------|---------|-----------|---------------------|
| Sinusoidal | Original Transformer | Add to embeddings | Moderate |
| Learned | GPT-2, BERT | Trainable addition | Poor |
| RoPE | Llama 2/3, Mistral | Rotate Q and K | Good |
| ALiBi | BLOOM, MPT | Bias attention scores | Good |
| NoPE | Some recent models | None | N/A |

---

## 4. The Attention Mechanism

### 4.1 Intuition

#### The Core Problem Attention Solves

When reading "The cat sat on the mat **because it** was tired", you instantly know **"it" refers to "the cat"**, not "the mat". Your brain **attends** to the relevant part of the input. Attention gives neural networks the same ability.

```
┌─────────────────────────────────────────────────────────────────────┐
│     SELF-ATTENTION: "it" attends to other words                     │
│                                                                     │
│  "The cat sat on the mat because it was tired"                      │
│                                                                     │
│   The  cat  sat  on  the  mat  because  it  was  tired             │
│    │    │    │   │    │    │      │      │   │    │                │
│    │    │    │   │    │    │      │      │   │    │                │
│    │    ├────┼───┼────┼────┼──────┤      │   │    │                │
│    │    │    │   │    │    │      │      │   │    │                │
│    │    │    │   │    │    │  weight=0.72│   │    │  ← "it" strongly│
│    │    └────┴───┴────┴────┴──────┘      │   │    │    attends to   │
│    │                                     │   │    │    "cat"!       │
│                                                                     │
│  HOW IT WORKS:                                                      │
│  ─────────────                                                      │
│  Query (Q): "it" asks: "What do I refer to?"                       │
│  Key   (K): Each word has a label describing its content            │
│             "cat" key = [noun, animate, subject, ...]               │
│  Value (V): Actual content to extract:                              │
│             "cat" val = [furry, animal, subject, ...]               │
│                                                                     │
│  score = Q("it") · K("cat") / sqrt(d_k) = HIGH match!             │
│  Result: "it" gets enriched with info from "cat"                   │
│          → model resolves the pronoun!                              │
└─────────────────────────────────────────────────────────────────────┘
```

#### The Library Analogy

```
LIBRARY ANALOGY FOR Q, K, V:

  Query (Q):  You enter the library and ask:
              "I need information about machine learning."
              → What you're LOOKING FOR

  Key (K):    Each book's label/tag on the shelf:
              "ML Fundamentals", "Cooking Recipes", "Neural Networks"
              → What each source CONTAINS

  Value (V):  The actual CONTENT inside each book
              → The information you'll EXTRACT

  Attention:  You compare your query with each key to find the best
              matches, then read the corresponding values.
              "ML Fundamentals" and "Neural Networks" match well
              → You read those books more carefully.
```

---

### 4.2 The Attention Formula

```
                     ┌──────────────────────────────────────────┐
                     │                                          │
                     │         QK^T                            │
                     │  softmax(─────) V                       │
                     │          √d_k                           │
                     │                                          │
                     └──────────────────────────────────────────┘

Where:
  Q  = Query matrix    Shape: (seq_len, d_k)   "What am I looking for?"
  K  = Key matrix      Shape: (seq_len, d_k)   "What do I contain?"
  V  = Value matrix    Shape: (seq_len, d_v)   "What information do I have?"
  d_k = dimension of key vectors               (used for scaling)
```

#### Attention Mechanism Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                   ATTENTION MECHANISM FLOW                           │
│                                                                      │
│    Input Embeddings X                                                │
│           │                                                          │
│    ┌──────┼──────┐                                                  │
│    │      │      │                                                   │
│   W_Q    W_K    W_V   (learned weight matrices)                     │
│    │      │      │                                                   │
│    Q      K      V                                                   │
│    │      │      │                                                   │
│    └──────┘      │                                                   │
│       Q × K^T   │                                                   │
│         │        │                                                   │
│       ÷ √d_k    │   (scale to prevent gradient issues)             │
│         │        │                                                   │
│     Softmax      │   (convert scores to probabilities)              │
│         │        │                                                   │
│     Weights ×    V   (weighted combination of values)               │
│         │                                                            │
│      Output (context-aware representation)                           │
└──────────────────────────────────────────────────────────────────────┘
```

---



#### Attention Weight Heatmap (Conceptual)

```
Attention weights for "The cat sat on the mat because it was tired":
(Higher value = stronger attention)

Attending TO:    The  cat  sat  on  the  mat  because  it  was  tired
                ┌──────────────────────────────────────────────────────┐
"The"          │ .25  .15  .10  .05  .20  .10   .05   .05  .03  .02  │
"cat"          │ .10  .30  .20  .05  .05  .10   .08   .07  .03  .02  │
"sat"          │ .05  .25  .25  .10  .05  .10   .08   .07  .03  .02  │
"on"           │ .10  .10  .15  .20  .10  .15   .08   .07  .03  .02  │
"the"          │ .25  .10  .05  .05  .25  .10   .05   .05  .05  .05  │
"mat"          │ .05  .10  .15  .20  .10  .20   .08   .07  .03  .02  │
"because"      │ .08  .15  .12  .05  .08  .12   .20   .10  .05  .05  │
"it"           │ .05  .72  .05  .02  .05  .02   .05   .02  .01  .01  │  ← "it" attends
"was"          │ .05  .40  .10  .05  .05  .05   .10   .10  .05  .05  │     strongly to
"tired"        │ .05  .35  .10  .05  .05  .05   .10   .15  .05  .05  │     "cat"!
               └──────────────────────────────────────────────────────┘
```

---

### 4.3 Why Scale by √dk?

```
The Problem: Without scaling, dot products grow with d_k

  d_k = 4:    dot products have std ≈ 2
  d_k = 64:   dot products have std ≈ 8
  d_k = 512:  dot products have std ≈ 23
  d_k = 4096: dot products have std ≈ 64

For d_k=4096, attention scores range from roughly -192 to +192.

What happens to softmax with extreme values?
  softmax([1, 2, 3])         = [0.09, 0.24, 0.67]  ← spread out, good 
  softmax([10, 20, 30])      = [0.00, 0.00, 1.00]  ← nearly one-hot 
  softmax([100, 200, 300])   = [0.00, 0.00, 1.00]  ← completely saturated 

One-hot softmax → gradient ≈ 0 → model stops learning!
```

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Demonstrate softmax behavior:
x = np.array([1.0, 2.0, 3.0])
print("Softmax behavior with different scales:")
print(f"  softmax([1, 2, 3]):       {softmax(x)}")        # Reasonable distribution
print(f"  softmax([10, 20, 30]):    {softmax(x * 10)}")   # Nearly one-hot!
print(f"  softmax([100, 200, 300]): {softmax(x * 100)}") # Completely saturated


# Show that std of dot products grows as sqrt(d_k)
np.random.seed(42)
for d_k in [4, 64, 512, 4096]:
    Q = np.random.randn(10000, d_k)
    K = np.random.randn(10000, d_k)
    dot_products = np.sum(Q * K, axis=1)
    print(f"d_k = {d_k:5d}: std = {dot_products.std():8.3f}, "
          f"std/sqrt(d_k) = {dot_products.std()/np.sqrt(d_k):.3f}")

# d_k =     4: std =    2.011, std/sqrt(d_k) = 1.005
# d_k =    64: std =    7.989, std/sqrt(d_k) = 0.999
# d_k =   512: std =   22.593, std/sqrt(d_k) = 0.999
# d_k =  4096: std =   63.938, std/sqrt(d_k) = 0.999

# SOLUTION: Divide by sqrt(d_k) → variance always ~1 → softmax well-behaved!
```

---


#### Multi-Head Attention (Brief Overview)

```
┌─────────────────────────────────────────────────────────────────────┐
│                 MULTI-HEAD ATTENTION                                 │
│                                                                     │
│  Instead of one attention function with d_model dimensions,         │
│  run h attention functions in parallel with d_model/h each.         │
│                                                                     │
│  Input X                                                            │
│     │                                                               │
│  ┌──┼──┬──┬──┐   Split into h heads                               │
│  │  │  │  │  │                                                      │
│ H1 H2 H3 H4 H5  ...Hh   ← each head attends differently          │
│  │  │  │  │  │                                                      │
│  └──┼──┴──┴──┘   Concatenate outputs                               │
│     │                                                               │
│   W_O projection                                                    │
│     │                                                               │
│  Final output                                                       │
│                                                                     │
│  Why multi-head?                                                    │
│  Head 1: might focus on syntax     (subject-verb agreement)        │
│  Head 2: might focus on coreference ("it" → "cat")                 │
│  Head 3: might focus on proximity  (nearby words)                  │
│  Head 4: might focus on semantics  (related meanings)              │
│  ...                                                                │
│  Different heads specialize in different relationship types!        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Week 2 Summary

###  Key Takeaways

| Concept | One-Line Summary |
|---------|-----------------|
| **Tokenization** | Converts text to integers using BPE subword splitting — balances vocabulary size with sequence length |
| **Embeddings** | Maps tokens to dense vectors where semantic similarity = geometric proximity |
| **Contextual Embeddings** | Same word gets different representations based on surrounding context (BERT, GPT) |
| **Positional Encodings** | Injects position info since transformers process all tokens in parallel |
| **RoPE** | State-of-the-art positional encoding: rotates Q and K vectors by position-dependent angles |
| **Self-Attention** | Each token computes a weighted combination of all other tokens via Q·K^T/√dk |
| **Scaling by √dk** | Prevents attention scores from growing large, which would saturate softmax and kill gradients |

### The Full Pipeline

```
"The cat sat on the mat because it was tired"
              │
              ▼
  ┌───────────────────────┐
  │    1. TOKENIZATION    │  "it" → token_id 428
  │    (BPE subwords)     │  Sequence: [464, 3797, 3332, 319, 262, 2603, 780, 340, 373, 10032]
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │    2. EMBEDDING       │  token_id 428 → [0.3, 0.7, 0.1, ..., 0.9]  (768 dims)
  │    (Lookup table)     │  All tokens become dense vectors
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │    3. POSITIONAL ENC  │  token_embed + pos_embed (or RoPE rotation)
  │    (Inject position)  │  Now each vector encodes WHAT + WHERE
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │    4. ATTENTION       │  "it" (Q) computes scores with all K vectors
  │    (Self-attend)      │  Highest score: "cat" → output enriched with cat info
  └───────────────────────┘
              │
              ▼
  Context-aware representations → Feed-Forward → Output
```

---

## Exercises

**1. BPE from Scratch: Implement the full BPE tokenizer and train it on a book from [Project Gutenberg](https://www.gutenberg.org/). How does vocabulary size affect compression ratio?**

---

See the implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\tokenization_vectorization_Attention\BPE_from_scratch.ipynb`

**Overview**

This project implements a Byte Pair Encoding (BPE) tokenizer from scratch in Python.
The tokenizer is trained on a book downloaded from Project Gutenberg and demonstrates how vocabulary size affects compression ratio.

BPE is widely used in modern language models to convert text into tokens efficiently.

**Objective**

1. Implement the BPE algorithm from scratch

2. Train the tokenizer on a real text dataset

3. Observe how changing vocabulary size changes compression ratio

**Dataset**

The training text is taken from the book Alice's Adventures in Wonderland from Project Gutenberg.

**How BPE Works**

1. Convert the text into raw bytes (0–255).

2. Count the most frequent pair of tokens.

3. Merge the pair into a new token.

4. Repeat until the desired vocabulary size is reached.

This process creates subword tokens that represent common patterns in the text.

**Compression Ratio**

Compression ratio measures how much the token count decreases after BPE.

Compression Ratio =
Original Tokens / Tokens After BPE

Example:

Original tokens: 100000

Tokens after BPE: 60000

Compression ratio = 1.67

This means the text became 1.67× smaller in tokens.

**Effect of Vocabulary Size**

**| Vocabulary Size --> 	Result |**


| Small     --> 	Many tokens, low compression |

| Medium  --> 	Balanced tokenization |

| Large	-->  Fewer tokens, better compression |

Increasing vocabulary size allows more merges, which reduces the number of tokens.

**Technologies Used**

1. Python

2. Google Colab

3. Collections library (Counter)

4. Matplotlib (for visualization)

**Key Learning**

1. How tokenizers used in LLMs work

2. Why subword tokenization is important

3. Relationship between vocabulary size and compression

**How to Run**

1. Open the notebook in Google Colab.

2. Download the dataset from Project Gutenberg.

3. Run all cells to train the BPE tokenizer.

4. Observe the compression ratio for different vocabulary sizes.

**Conclusion**

BPE improves text representation by merging frequent token pairs.

Larger vocabulary sizes generally lead to better compression, but they also increase model memory and computation requirements.

---

**2. Tokenizer Comparison: Use `tiktoken` to tokenize the same text with GPT-2 (`r50k_base`) vs GPT-4 (`cl100k_base`) tokenizers. Which produces fewer tokens? Why?**

---
See the implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\tokenization_vectorization_Attention\tokenizer_comparison.ipynb`

**Tokenizer Comparison: GPT-2 vs GPT-4**

**Project Goal**

This project compares two tokenizers using the **tiktoken** library:

* GPT-2 tokenizer (`r50k_base`)
* GPT-4 tokenizer (`cl100k_base`)

The same text is tokenized with both tokenizers to see which produces fewer tokens.

**Tools Used**

* Python
* tiktoken library
* Google Colab


**What is Tokenization?**

Tokenization is the process of breaking text into smaller pieces called tokens so that language models can understand and process the text.

Example:

Text:
I love machine learning

Tokens:
I | love | machine | learning

**Experiment**

1. write a text.
2. Load GPT-2 and GPT-4 tokenizers.
3. Convert the text into tokens.
4. Count the number of tokens for each tokenizer.
5. Compare the results.

**Result**

The GPT-4 tokenizer (`cl100k_base`) usually produces fewer tokens than the GPT-2 tokenizer (`r50k_base`).

Example result:

* GPT-2 tokens: 37,700
* GPT-4 tokens: 29,500

**Why GPT-4 Produces Fewer Tokens**

* GPT-4 tokenizer has a larger vocabulary (~100k tokens).
* It can represent whole words or larger subwords.
* GPT-2 tokenizer has a smaller vocabulary (~50k tokens).
* Therefore GPT-4 splits text into fewer pieces.

Example:

Word: unbelievable

GPT-2:
un + bel + iev + able (4 tokens)

GPT-4:
unbelievable (1 token)

**Conclusion**

The GPT-4 tokenizer is more efficient because it produces fewer tokens for the same text, which helps models process text more efficiently.

**How to Run**

1. Open the notebook in Google Colab.
2. Install the library:
   pip install tiktoken
3. Run the code cells to compare token counts.

---
**3. Word2Vec Analogies: Train Word2Vec on a larger corpus (try Wikipedia or Common Crawl) and find interesting analogies beyond "king - man + woman = queen". What other semantic relationships can you discover?**

---


See implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\tokenization_vectorization_Attention\word2vec_analogies.ipynb`

**What is this project**

This project shows how to train a Word2Vec model using a large text dataset and test how well it understands relationships between words (like king → queen).

**What is Word2Vec?**

Word2Vec is a technique that converts words into vectors (numbers) so that:

* Similar words are close together

* Relationships between words can be captured mathematically

Example:

king - man + woman ≈ queen

**Steps in the Code**

1. Install required library

2. Import libraries

3. . Load dataset

dataset = api.load("text8")

text8 = cleaned Wikipedia text

Already tokenized (words are separated)

4. Train Word2Vec model

* vector_size=100 → each word becomes a vector of size 100

* window=5 → looks at 5 words before and after

* min_count=5 → ignores rare words

* workers=4 → uses 4 CPU cores (faster training)

**Testing the Model**

Example 1: Gender relationship

`model.wv.most_similar(positive=["king", "woman"], negative=["man"])`

Expected: queen

**What does this show**

The model learns:

Gender relationships → king → queen

**Key Takeaways**

* Word2Vec learns meaning from context

* Words become vectors

* You can do math on words

* Useful for NLP tasks like:

* Similarity search

* Recommendation systems

* Text classification

* Words that appear in similar contexts have similar meanings.

---
**4. Self-Attention from Scratch: Implement the full self-attention formula in NumPy, then verify your results match PyTorch's `nn.MultiheadAttention` for the same inputs.**

---

See implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\tokenization_vectorization_Attention\self_attention_from_scratch.ipynb`

**Self-Attention: NumPy vs PyTorch (Simple Explanation)**

**What this code does**

This project shows how self-attention works in two ways:

* Using NumPy (from scratch)

* Using PyTorch built-in MultiheadAttention

Then it compares both outputs to check if they match.

**Idea in simple words**

Self-attention helps a model understand:

“Which words (or tokens) are important compared to others?”

It does this using 3 things:

* Q (Query) → What I am looking for

* K (Key) → What I contain

* V (Value) → What I give

**Steps in NumPy Self-Attention**

1. Convert input X into:

    * Q = X × Wq

    * K = X × Wk

    * V = X × Wv

2. Compute similarity:

      scores = Q × Kᵀ

3. Scale it:

      scores / √dk

4. Apply softmax → gives attention weights

5. Final output:

     output = weights × V

**PyTorch Version**

* Uses nn.MultiheadAttention

* We manually set:

     Query, Key, Value weights

* Disable extra projection to match NumPy

* Run attention on same input

**Why transpose weights in PyTorch**

PyTorch stores weights differently, so we use:

          W.T

to match NumPy calculations.

**Final Comparison**

We compute:

       mean difference between NumPy and PyTorch output

If everything is correct:

      The difference should be very small (close to 0)

**Key Takeaways**

* Self-attention = compare tokens + combine information

* NumPy helps understand the math

* PyTorch makes it fast and scalable

* Both should give same result if implemented correctly

* Helps understand how Transformers work internally

* Builds strong intuition for LLMs



---
**5. Causal Masking: Modify the attention implementation to use a causal mask. How do attention patterns change? Can you visualize where each token attends?**

---

See implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\tokenization_vectorization_Attention\casual_masking.ipynb`

**Self-Attention (With & Without Mask)**

This project shows how self-attention works in a very simple way using PyTorch.

We compute attention:

* Without mask (can see all words)

* With causal mask (can only see past words)

**What is Self-Attention**

Self-attention helps a model understand:

“When I read this word, which other words should I focus on”

Example:

"I love learning AI"

When reading "learning", the model may focus on:

* "love"

* "AI"

**Steps in the Code**

1. Create Input

We start with tokens:

      tokens = ["I","love","learning","AI"]

Convert them into vectors (embeddings):

      X = torch.randn(seq_len, d_model)

2. Create Q, K, V

We compute:

* Query (Q) → what I am looking for

* Key (K) → what I contain

* Value (V) → actual information

`Q = X @ Wq,
K = X @ Wk,
V = X @ Wv`

3. Compute Attention Scores

         scores = Q @ K.T / sqrt(d_model)

This tells how much each word relates to others.

4. Apply Softmax

         attention = softmax(scores)

Now values become probabilities (0 to 1).

**Attention WITHOUT Mask**

* Each word can see all words

* Used in models like encoders

* Output: Heatmap shows full connections

* Example:

"AI" can look at "I", "love", "learning"

**Attention WITH Causal Mask**

We block future words using a mask:

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

Then apply:

        masked_scores = scores + mask

**Why Mask**

Because in tasks like text generation:

A word should NOT see future words

Example:

"learning" should NOT see "AI"

**After Masking**

Each word sees only:

* itself

* previous words

* Heatmap becomes triangular

**Visualization**

We use:

* matplotlib

* seaborn

To plot attention heatmaps:

* Blue → no mask

* Red → causal mask

**Key Takeaways**

* Self-attention finds relationships between words

* Q, K, V are core components

* Softmax converts scores → probabilities

* Masking prevents cheating (seeing future)

* This is the core idea behind Transformers used in:

ChatGPT, BERT, GPT models



---

## Further Reading

| Resource | Topic |
|----------|-------|


| ["Effective Approaches to Attention-based NMT" — Luong et al. (2015)](https://arxiv.org/abs/1508.04025) | Attention foundations | 

| ["RoFormer: Enhanced Transformer with RoPE" — Su et al. (2021)](https://arxiv.org/abs/2104.09864) | Rotary embeddings | 

| [HuggingFace Tokenizers Documentation](https://huggingface.co/docs/tokenizers) | Practical tokenizer usage | 

| ["The Illustrated Transformer" — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Visual walkthrough | 


---


**1.["Effective Approaches to Attention-based NMT" — Luong et al. (2015)](https://arxiv.org/abs/1508.04025)**

---

**Authors:** Thang Luong, Hieu Pham, Christopher D. Manning

**What This Paper Does**

Luong et al., 2015 improves machine translation by introducing attention. This allows the translator (decoder) to focus on the right source words for each word it translates, instead of relying on a single vector for the whole sentence.

**Main Ideas**

* Problem: Old NMT uses one vector → loses details, especially for long sentences.

* Solution: Attention mechanism to focus on relevant words.

* Types of Attention:

* Global: looks at all source words.

* Local: looks at a small window of source words.

* Scoring Methods: dot, general, concat.

* Input Feeding: previous attention outputs are fed into the next decoder step for better context.

**Example Diagram**

```
Source: I   love   cats
Encoder states: h1  h2  h3


Decoder generates target: J'  aime  les  chats
- Global: looks at h1, h2, h3 for each target word
- Local: looks at a small window around predicted alignment
- Attention weights determine focus for each output word
```

**Benefits**

* Handles long sentences well.

* Produces more accurate translations.

* Can visualize which source words affect each translated word.

**Drawbacks**

* Slightly slower than plain encoder-decoder.

* Local attention may miss distant context.

* Needs good training data.

**Why It’s Important**

* First practical way to implement attention in RNN-based NMT.

* Inspired Transformers and modern attention-based models.

**Easy Analogy**

* Without attention: Summarize the whole story in 1 sentence → lose details.

* With attention: Look at relevant parts for each translated word → accurate translation.

---

**2.["RoFormer: Enhanced Transformer with RoPE" — Su et al. (2021)](https://arxiv.org/abs/2104.09864) | Rotary embeddings |**

---

**1. What is this paper about**

* Transformers don’t understand word order by default.

* Older methods add position info, but they don’t handle long sequences well.

* This paper introduces RoPE (Rotary Position Embedding) to fix this.

* It encodes position by rotating vectors inside attention instead of adding them.

**2. Key Idea (Very Simple)**

* Split vector into pairs → treat each pair like a 2D point

* Rotate each pair using sin & cos based on position

* Use rotated vectors in attention

* Result: attention now understands relative distance between words

**3. How it Works**

* Take Query (Q) and Key (K)

* Apply rotation based on position

* Compute attention normally

`Q, K → Rotate → Attention → Output`

**4. Advantages**

* Captures relative + absolute position together

* Works for long sequences (no fixed limit)

* Far tokens have less influence (natural decay)

* Works with efficient transformers

* Slightly better performance on long text tasks

**5. Disadvantages**

* Slightly more complex math

* Adds small computation overhead

* Improvement is sometimes not huge

**6. Intuition**

Think like this:

* Each token = arrow

* RoPE = rotate arrow based on position

* Attention = compares rotated arrows

* So model understands:

“how far words are”

“which words are closer or important”

**7. Why it is Important**

* Solves limitation of old positional encodings

* Works well for long-context models (like LLMs)

* Now widely used in modern models

`RoPE = Rotate vectors → Encode position inside attention → Better understanding of word order`

---
**3. | [HuggingFace Tokenizers Documentation](https://huggingface.co/docs/tokenizers) | Practical tokenizer usage |**

---

**HuggingFace Tokenizers (Simple README)**

**What is this**

HuggingFace Tokenizers is a **library** that converts text into numbers (tokens) so AI models can understand it.

Example:

```
"I love AI" → ["I", "love", "AI"] → [101, 234, 567]
```



**Why do we need it**

* Models **cannot understand text** directly
* They only understand **numbers**
* Tokenizer acts like a **translator (text → numbers)**



**How it works**

Tokenizer is not just splitting words, it has steps:

1. **Normalizer** → cleans text ("HELLO" → "hello")
2. **Pre-tokenizer** → splits text
3. **Model** → creates subwords (BPE, WordPiece)
4. **Post-processing** → adds special tokens
5. **Decoder** → converts back to text



**Key Features**

*  Very fast (built in Rust)
*  Supports BPE, WordPiece, Unigram
*  Adds special tokens automatically
*  Handles padding & truncation
*  Keeps track of token positions

---

**Special Tokens**

* `[CLS]` → start of sentence
* `[SEP]` → separator
* `[PAD]` → padding
* `[UNK]` → unknown words



**Padding & Truncation**

* **Padding** → make all inputs same length
* **Truncation** → cut long text



**Vocabulary**

* Tokenizer creates a **dictionary (vocab)**
* Each word/subword gets a number
* Unknown words are split into smaller parts



**Why it is important**

* First step in every NLP/LLM pipeline
* Affects model performance
* Impacts speed and cost


**Advantages**

* Very fast
* Flexible
* Easy to use with models
* Handles large data


**Disadvantages**

* Can be confusing for beginners
* Many options/settings
* Must match tokenizer with model



**Final Idea**

Tokenizer = bridge between **text and AI model**

Without tokenizer → model cannot understand anything.

---

**4. | ["The Illustrated Transformer" — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Visual walkthrough |**

---

**The Illustrated Transformer (Jay Alammar)**

**What is it**

* Not a research paper
* A **visual blog guide** explaining the Transformer model
* Makes a complex topic **easy using diagrams**



**Why it became popular**

* Original Transformer paper was **hard to understand**
* This guide explains it in a **simple, step-by-step way**
* Helped many people learn **how GPT/BERT work**



**What is inside**

**Big Idea**

Transformer = model that uses **attention** instead of RNN/CNN


**Main Steps**

1. **Input Embedding**

   * Words → numbers (vectors)

2. **Positional Encoding**

   * Adds word order information

3. **Self-Attention (Core)**

   * Each word looks at other words
   * Decides what is important

4. **Query, Key, Value**

   * Query → what I want
   * Key → what I match
   * Value → actual info

5. **Multi-Head Attention**

   * Multiple attentions at once
   * Learns different patterns

6. **Feed Forward Network**

   * Small neural network per word

7. **Add & Normalize**

   * Helps stable training

8. **Decoder**

   * Generates output word-by-word
   * Uses masking (no future words)

9. **Final Output**

   * Predict next word using Softmax



**Key Takeaways**

* Uses **attention instead of sequence processing**
* Processes all words **in parallel**
* Captures **long-distance relationships**
* Foundation of modern LLMs



**Advantages**

* Faster than RNN (parallel processing)
* Understands long context well
* Scales to large data
* Works for text, images, code



**Disadvantages**

* High computation cost (O(n²))
* Needs large data
* Needs positional encoding
* Hard to fully interpret



**Simple Intuition**

Old models:

```
Read words one by one (slow)
```

Transformer:

```
Looks at all words together (smart & fast)
```

---

**Final Summary**

* Best beginner-friendly explanation of Transformers
* Focuses on self-attention (main idea)
* Helps understand GPT, BERT, and LLMs
* Must-learn if you want to study modern AI

---


