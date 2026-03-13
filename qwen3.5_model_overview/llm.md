# Large Language Models (LLMs)
**Author:** Soumili Jana Ghosh


---

## Table of Contents

1. [What is an LLM?](#1-what-is-an-llm)
2. [Brief History](#2-brief-history)
3. [High-Level Architecture Overview](#3-high-level-architecture-overview)
4. [Step-by-Step Data Flow](#4-step-by-step-data-flow)
   - 4.1 [Tokenization](#41-tokenization)
   - 4.2 [Token Embeddings](#42-token-embeddings)
   - 4.3 [Positional Encoding](#43-positional-encoding)
   - 4.4 [The Transformer Block](#44-the-transformer-block)
   - 4.5 [Multi-Head Self-Attention](#45-multi-head-self-attention)
   - 4.6 [Feed Forward Network](#46-feed-forward-network)
   - 4.7 [Layer Normalization & Residual Connections](#47-layer-normalization--residual-connections)
   - 4.8 [Stacking Layers](#48-stacking-layers)
   - 4.9 [The Output Head (LM Head)](#49-the-output-head-lm-head)
   - 4.10 [Autoregressive Generation](#410-autoregressive-generation)
5. [Inside Attention — The Full Picture](#5-inside-attention--the-full-picture)
   - 5.1 [Q, K, V — Query, Key, Value](#51-q-k-v--query-key-value)
   - 5.2 [Scaled Dot-Product Attention](#52-scaled-dot-product-attention)
   - 5.3 [Multi-Head Attention](#53-multi-head-attention)
   - 5.4 [Causal (Masked) Attention](#54-causal-masked-attention)
   - 5.5 [KV Cache (Inference Optimization)](#55-kv-cache-inference-optimization)
6. [Training an LLM](#6-training-an-llm)
   - 6.1 [Pre-Training](#61-pre-training)
   - 6.2 [Supervised Fine-Tuning (SFT)](#62-supervised-fine-tuning-sft)
   - 6.3 [RLHF](#63-rlhf--reinforcement-learning-from-human-feedback)
   - 6.4 [Constitutional AI / RLAIF](#64-constitutional-ai--rlaif)
   - 6.5 [Loss Functions](#65-loss-functions)
7. [Key Parameters & Hyperparameters](#7-key-parameters--hyperparameters)
8. [Context Window & Memory](#8-context-window--memory)
9. [Tokenizer Deep Dive](#9-tokenizer-deep-dive)
10. [Embeddings Deep Dive](#10-embeddings-deep-dive)
11. [Inference & Decoding Strategies](#11-inference--decoding-strategies)
12. [Model Sizes & Scaling Laws](#12-model-sizes--scaling-laws)
13. [Popular LLM Architectures](#13-popular-llm-architectures)
14. [What LLMs Can & Cannot Do](#14-what-llms-can--cannot-do)
15. [Limitations & Failure Modes](#15-limitations--failure-modes)
16. [Hardware: GPUs, TPUs & Memory](#16-hardware-gpus-tpus--memory)
17. [The Future of LLMs](#17-the-future-of-llms)
18. [Complete Glossary](#18-complete-glossary)

---

## 1. What is an LLM?

**LLM = Large Language Model**

| Word | Meaning |
|------|---------|
| **Large** | Trained on trillions of words; has billions (sometimes trillions) of parameters |
| **Language** | Works with human text: English, code, math notation, JSON, etc. |
| **Model** | A mathematical function that maps input → output |

### What does it actually do?

At its core, an LLM does one thing:

```
Given the words so far → predict the most likely next word
```

When this approach is applied at a very large scale with high-quality training and large computational resources, the system becomes highly capable. It can perform many tasks such as writing essays, debugging code, answering questions, translating languages, and reasoning through complex problems.

### The simplest mental model

A simple way to think about it is like the autocomplete feature on your phone. It predicts the next word you might type. An LLM does the same thing, but it is trained on a massive amount of information and has much more computing power.

---

## 2. Brief History

```
1950s   ── Alan Turing proposes the "Turing Test" for machine intelligence
1980s   ── Recurrent Neural Networks (RNNs) appear; struggle with long sequences
2013    ── Word2Vec: words represented as vectors (embeddings) for the first time
2015    ── LSTMs improve sequential text processing; still slow and limited
2017    ── "Attention Is All You Need" paper introduces the Transformer
2018    ── BERT (Google) — bidirectional Transformer; GPT-1 (OpenAI)
2019    ── GPT-2 (OpenAI) — 1.5B params; OpenAI delays release fearing misuse
2020    ── GPT-3 — 175B parameters; shocks the world with few-shot learning
2022    ── ChatGPT launched — 100M users in 60 days; fastest product in history
2023    ── GPT-4, Claude, Gemini; LLaMA open-sourced; the LLM explosion
2024    ── Multimodal models, reasoning models (o1), 1M+ token context windows
2025+   ── Agentic AI, on-device models, AI that takes real-world actions
```

---

## 3. High-Level Architecture Overview

Here is the complete flow from your text input to the model's text output:

```
╔══════════════════════════════════════════════════════════════════════╗
║                        LLM ARCHITECTURE                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  INPUT TEXT:  "What is the capital of France?"                       ║
║                         │                                            ║
║                         ▼                                            ║
║  ┌─────────────────────────────────┐                                 ║
║  │        TOKENIZER                │  "What", " is", " the",        ║
║  │  Text → Token IDs               │  " capital", " of",            ║
║  └─────────────────────────────────┘  " France", "?"                ║
║                         │                                            ║
║                         ▼                                            ║
║  ┌─────────────────────────────────┐                                 ║
║  │     TOKEN EMBEDDING TABLE       │  IDs → Dense Vectors           ║
║  │  ID → High-Dimensional Vector   │  [0.2, -0.5, 0.8, ...]        ║
║  └─────────────────────────────────┘                                 ║
║                         │                                            ║
║                         ▼                                            ║
║  ┌─────────────────────────────────┐                                 ║
║  │     POSITIONAL ENCODING         │  Add position info             ║
║  │  "Which position is this token?"│  (word 1? word 5? word 100?)   ║
║  └─────────────────────────────────┘                                 ║
║                         │                                            ║
║           ┌─────────────┴─────────────┐                             ║
║           │  TRANSFORMER BLOCK × N    │  (N = 12 to 96+ layers)    ║
║           │  ┌─────────────────────┐  │                             ║
║           │  │ Multi-Head          │  │                             ║
║           │  │ Self-Attention      │  │  "Which words should I      ║
║           │  └─────────────────────┘  │   pay attention to?"       ║
║           │           +               │                             ║
║           │  ┌─────────────────────┐  │                             ║
║           │  │ Feed Forward        │  │  "Process and transform     ║
║           │  │ Network             │  │   the information"          ║
║           │  └─────────────────────┘  │                             ║
║           │           +               │                             ║
║           │  Layer Norm + Residuals   │                             ║
║           └─────────────┬─────────────┘                             ║
║                         │  (repeated N times)                       ║
║                         ▼                                            ║
║  ┌─────────────────────────────────┐                                 ║
║  │       FINAL LAYER NORM          │                                 ║
║  └─────────────────────────────────┘                                 ║
║                         │                                            ║
║                         ▼                                            ║
║  ┌─────────────────────────────────┐                                 ║
║  │        LM HEAD (Linear Layer)   │  Vector → Vocab Scores         ║
║  │  Vector → Logits over vocab     │  50,000+ scores                ║
║  └─────────────────────────────────┘                                 ║
║                         │                                            ║
║                         ▼                                            ║
║  ┌─────────────────────────────────┐                                 ║
║  │          SOFTMAX                │  Scores → Probabilities        ║
║  └─────────────────────────────────┘                                 ║
║                         │                                            ║
║                         ▼                                            ║
║  ┌─────────────────────────────────┐                                 ║
║  │       DECODING STRATEGY         │  Pick the next token           ║
║  │  (Greedy / Top-k / Nucleus)     │                                 ║
║  └─────────────────────────────────┘                                 ║
║                         │                                            ║
║                         ▼                                            ║
║  OUTPUT:  "Paris"  (then repeats until done)                         ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 4. Step-by-Step Data Flow

### 4.1 Tokenization

Before the model can read your text, it must convert it into numbers. This is done by a **tokenizer**.

#### What is a Token?

A token is the basic unit of text. It is NOT always a full word:

```
"Hello"           → 1 token
"unbelievable"    → ["un", "believ", "able"]       → 3 tokens
"ChatGPT"         → ["Chat", "G", "PT"]            → 3 tokens
"1234"            → ["12", "34"]                   → 2 tokens
"   " (spaces)    → 1 token
"\n" (newline)    → 1 token
```

A rough rule: **1 token ≈ 0.75 English words** (or ~4 characters).

#### How Tokenization Works (BPE — Byte Pair Encoding)

Most LLMs use **BPE (Byte Pair Encoding)**:

```
Step 1: Start with individual characters: ["h","e","l","l","o"]
Step 2: Merge the most common character pairs: "he", "ll", "o" → ["he", "ll", "o"]
Step 3: Merge again: "hel", "lo" → ["hel", "lo"]
Step 4: After enough merges: "hello" → ["hello"] → 1 token
```

The tokenizer was trained on a massive corpus and knows which combinations are most common.

#### Token ID Assignment

Each token gets a unique integer ID:

```
Token         →  ID
"Hello"       →  15496
" world"      →  995
"!"           →  0
"The"         →  464
" capital"    →  3139
```

These IDs are what actually go into the model.

---

### 4.2 Token Embeddings

Raw IDs like `[15496, 995, 0]` are just numbers — they don't capture meaning. The model converts each ID into a rich vector (list of numbers) using a large lookup table called the **embedding matrix**.

```
Embedding Matrix (simplified):

Token ID  →  Embedding Vector (d_model dimensions)
─────────────────────────────────────────────────────
  464     →  [0.21, -0.45,  0.87, -0.12,  0.63, ...]  ← "The"
 3139     →  [0.15, -0.31,  0.72, -0.08,  0.91, ...]  ← "capital"
 1659     →  [-0.88, 0.19, -0.33,  0.77, -0.20, ...]  ← "dog"
```

**Why vectors?** Because:
- Similar words end up with similar vectors (close in vector space)
- Relationships between words are encoded in the directions

Famous example:
```
vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen")
```

The embedding dimension `d_model`(the size of the vector used to represent each token inside a Transformer model) is typically:
- Small models: 768
- Medium models: 1024–2048
- Large models: 4096–12288

---

### 4.3 Positional Encoding

The Transformer processes all tokens **simultaneously** (in parallel), unlike RNNs which go one word at a time. This is faster, but it means the model loses track of **word order**.

The fix: **add position information** to each embedding before processing.

```
Token 1: "The"     embedding + position_1_vector
Token 2: "capital" embedding + position_2_vector
Token 3: "of"      embedding + position_3_vector
Token 4: "France"  embedding + position_4_vector
```

There are two approaches:

#### Sinusoidal Positional Encoding (Original Transformer)
Uses sine and cosine waves at different frequencies. Each position gets a unique pattern:

```
Position 1: [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
Position 2: [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
```

#### RoPE — Rotary Positional Embedding (Modern LLMs)
Used by LLaMA, GPT-NeoX, Qwen and most modern models. Instead of adding a vector, it *rotates* the query and key vectors. This allows the model to generalize to longer sequences than it was trained on.

---

### 4.4 The Transformer Block

The heart of the LLM. A **Transformer Block** (also called a layer) takes the embeddings and refines them. Most LLMs stack many of these blocks:

```
Input (from previous layer or embedding)
        │
        ▼
┌───────────────────────────────────────┐
│         Layer Norm 1                  │  ← Normalize values
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    Multi-Head Self-Attention          │  ← "Which words matter?"
└───────────────────────────────────────┘
        │
        + ◄─── Residual Connection (add input back in)
        │
        ▼
┌───────────────────────────────────────┐
│         Layer Norm 2                  │  ← Normalize again
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    Feed Forward Network (MLP)         │  ← "Process and transform"
└───────────────────────────────────────┘
        │
        + ◄─── Residual Connection (add input back in)
        │
        ▼
Output (to next layer)
```

---

### 4.5 Multi-Head Self-Attention

This is the most important — and most misunderstood — part of the Transformer. See [Section 5.3](#5-inside-attention--the-full-picture) for the full deep dive.

**Simple version:** Each token looks at every other token and asks:
*"How much should I pay attention to you when computing my meaning?"*

---

### 4.6 Feed Forward Network

After attention, each token's representation passes through a small neural network independently. This is two linear layers with an activation function:

```
Input vector (d_model = 4096)
        │
        ▼
Linear Layer: 4096 → 16384     ← expands to 4× size
        │
        ▼
Activation Function (GELU/ReLU/SiLU)   ← adds non-linearity
        │
        ▼
Linear Layer: 16384 → 4096     ← compresses back
        │
        ▼
Output vector (d_model = 4096)
```

**What does this do?** While attention figures out *relationships between tokens*, the FFN processes each token's information and stores *factual knowledge*. Research shows that factual associations (like "Paris is the capital of France") are largely stored in the FFN weights.

---

### 4.7 Layer Normalization & Residual Connections

#### Layer Normalization
Keeps the numbers in a healthy range so training doesn't explode or vanish:

```
Before norm: [1200.5, -450.2, 8800.1, -2.3]   ← too wild
After norm:  [0.42,   -0.16,  0.89,  -0.01]   ← stable
```

Modern LLMs use **Pre-Layer Norm** (normalize before attention/FFN, not after) because it trains more stably.

#### Residual Connections (Skip Connections)
After each sub-layer (attention or FFN), the original input is **added back**:

```
output = layer(input) + input
```

Why? Two major benefits:
1. **Prevents vanishing gradients** — gradients can flow directly backward through many layers
2. **Preserves information** — early representations aren't overwritten, just refined

Without residuals, models deeper than ~10 layers become nearly impossible to train.

---

### 4.8 Stacking Layers

The same Transformer Block is repeated N times:

```
Embedding + Position
        │
    ┌───┴───┐
    │ Block 1│  ← Low-level patterns (syntax, common phrases)
    └───┬───┘
        │
    ┌───┴───┐
    │ Block 2│  ← Slightly higher-level patterns
    └───┬───┘
        │
       ...
        │
    ┌───┴───┐
    │Block 32│  ← High-level semantics, reasoning, facts
    └───┬───┘
        │
    Final Norm
```

Deeper layers encode more abstract, higher-level concepts. Lower layers handle syntax; upper layers handle meaning and reasoning.

**Number of layers in real models:**

| Model      | Layers | d_model | Heads |
|------------|--------|---------|-------|
| GPT-2 Small | 12    | 768     | 12    |
| GPT-3      | 96     | 12288   | 96    |
| LLaMA 2 7B | 32     | 4096    | 32    |
| LLaMA 2 70B| 80     | 8192    | 64    |
| GPT-4 (est)| 120+   | 12800+  | 96+   |
| Qwen1.5 7B| 32 | 4096  | 32 |
| Qwen2.5 0.5B| 24| 1024 | 16/2 (Q/KV) |
| Qwen3.5 397B(17B active)| 60 | 4096 | 32/2 (Q/KV) |
---

### 4.9 The Output Head (LM Head)

After the final Transformer layer, we have a vector for each token position. To predict the *next* token, we take the last token's vector and pass it through a **linear projection** to get scores for every possible token in the vocabulary:

```
Last hidden state:  [0.2, -0.5, 0.8, ..., 0.1]   (d_model = 4096 numbers)
         │
         ▼
LM Head (Linear):   4096 → 50,257 (vocab size)
         │
         ▼
Logits:  [2.1, -0.3, 0.8, 4.5, -1.2, ..., 0.7]   (50,257 raw scores)
         │
         ▼
Softmax: [0.001, 0.0003, 0.008, 0.24, 0.0001, ..., 0.002]   (probabilities, sum = 1.0)
```

The token with the highest probability is the model's best guess for what comes next.

---

### 4.10 Autoregressive Generation

LLMs generate text **one token at a time**, each time feeding the output back as new input:

```
Step 1:
  Input:  "The capital of France is"
  Output: "Paris"  (highest probability token)

Step 2:
  Input:  "The capital of France is Paris"
  Output: "."

Step 3:
  Input:  "The capital of France is Paris."
  Output: "<END>"  (special end-of-sequence token → stop generating)

Final output: "Paris."
```

This is called **autoregressive** generation — each new token depends on all previous tokens.

The model never "plans ahead." It just predicts the single most likely next token at each step, and chains these together to form coherent sentences.

---

## 5. Inside Attention — The Full Picture

### 5.1 Q, K, V — Query, Key, Value

Attention uses three concepts borrowed from information retrieval:

| Symbol | Name | Analogy | Role |
|--------|------|---------|------|
| **Q** | Query | "What am I looking for?" | The current token asking for information |
| **K** | Key | "What do I have?" | What each token advertises about itself |
| **V** | Value | "What do I actually give?" | The actual content to pass if selected |

Think of it like a **library search**:
- You have a **query** ("books about space")
- Books have **keys** ("astronomy", "physics", "history")
- You retrieve the **values** (the actual book content) of matching keys

In attention, the model learns these Q, K, V matrices during training.

```
For each token:
  Q = token_embedding × W_Q   ( the weight matrices are initialized with small random numbers, then learned during training.)
  K = token_embedding × W_K
  V = token_embedding × W_V
```

---

### 5.2 Scaled Dot-Product Attention

The attention score between two tokens is computed as:

```
               Q × Kᵀ
Attention =  ──────────  × V
               √d_k

Where:
  Q  = query matrix
  Kᵀ = transpose of key matrix
  d_k = dimension of keys (for scaling, helps stabilize training)
  V  = value matrix
```

**Step by step with an example:**

Sentence: `"The cat sat"`

```
Tokens: [The, cat, sat]

Step 1 — Compute Q, K, V for each token (via learned matrices)

Step 2 — Compute attention scores: Q × Kᵀ
         (how much does each token relate to every other token?)

         Scores matrix:
                  The    cat    sat
          The  [ 1.2,   3.4,   0.5 ]
          cat  [ 0.8,   4.1,   2.1 ]
          sat  [ 0.3,   2.8,   3.7 ]

Step 3 — Scale by √d_k (prevents values from getting too large)

Step 4 — Apply Softmax (convert to probabilities):
                  The    cat    sat
          The  [ 0.07,  0.87,  0.06 ]   ← "The" mostly attends to "cat"
          cat  [ 0.08,  0.83,  0.19 ]   ← "cat" strongly attends to itself
          sat  [ 0.04,  0.38,  0.58 ]   ← "sat" attends mostly to itself

Step 5 — Multiply by V to get weighted sum:
         Each token's output = weighted combination of all Value vectors
```

---

### 5.3 Multi-Head Attention

Instead of one attention operation, the model runs **multiple attention heads in parallel** — each with different Q, K, V weight matrices. This lets the model attend to different aspects of relationships simultaneously.

```
Input
  │
  ├──► Head 1 (Q₁, K₁, V₁)  →  Attention output 1   (syntax)
  ├──► Head 2 (Q₂, K₂, V₂)  →  Attention output 2   (coreference)
  ├──► Head 3 (Q₃, K₃, V₃)  →  Attention output 3   (semantics)
  ├──► Head 4 (Q₄, K₄, V₄)  →  Attention output 4   (position)
  │    ...
  └──► Head h (Qₕ, Kₕ, Vₕ)  →  Attention output h

Concatenate all outputs → Linear projection → Final output
```

Each head learns to focus on different types of relationships. Research has found heads that:
- Track subject-verb agreement
- Resolve pronoun references (it, he, they)
- Handle long-range dependencies
- Focus on adjacent tokens

A GPT-3 has **96 heads per layer × 96 layers = 9,216 attention heads** total.

---

### 5.4 Causal (Masked) Attention

During text generation, the model must not "cheat" by looking at future tokens. We use a **causal mask** (upper triangular mask) that blocks attention to future positions:

```
Sentence: [The, cat, sat, on, mat]

Attention mask (1 = allowed, 0 = blocked):

            The  cat  sat  on  mat
  The     [  1,   0,   0,   0,   0 ]  ← "The" can only see itself
  cat     [  1,   1,   0,   0,   0 ]  ← "cat" sees "The" + itself
  sat     [  1,   1,   1,   0,   0 ]  ← "sat" sees everything before it
  on      [  1,   1,   1,   1,   0 ]
  mat     [  1,   1,   1,   1,   1 ]  ← "mat" sees everything
```

This masking is applied during both training and inference, ensuring the model can only use past context to predict future tokens.

---

### 5.5 KV Cache (Inference Optimization)

During generation, the Keys and Values for already-generated tokens don't change. The **KV Cache** stores these so they don't need to be recomputed:

```
Without KV Cache:
  Step 1: Process ["The"]          → compute K,V for "The"
  Step 2: Process ["The","cat"]    → recompute K,V for "The" + compute for "cat"
  Step 3: Process ["The","cat","sat"] → recompute everything + compute for "sat"
  Cost: O(n²)  ← very slow for long sequences

With KV Cache:
  Step 1: Process ["The"]   → save K,V for "The" in cache
  Step 2: Process ["cat"]   → load cached K,V for "The", add new K,V for "cat"
  Step 3: Process ["sat"]   → load cache, add "sat"
  Cost: O(n)   ← much faster!
```

KV Cache is what makes real-time LLM responses possible. The trade-off: it uses significant GPU memory.

---

## 6. Training an LLM

### 6.1 Pre-Training

The foundational phase where the model learns from raw text. Pre-training means the model learns language patterns from a huge amount of text before being used for tasks. This table shows the different types of text used to train large language models. Each dataset contributes a certain number of tokens and has a specific weight during training.

#### Data
```
Pre-training data sources (approximate for GPT-3 class model):
  
  Source               Size        Weight
  ───────────────────────────────────────
  Common Crawl         410B tokens   60%    (web pages)
  WebText2             19B tokens     22%   (Reddit links)
  Books1 + Books2      67B tokens     8%    (fiction + non-fiction)
  Wikipedia             3B tokens     3%    (encyclopedic)
  GitHub               ???            ???   (code)
  Other                ...           ...
```

#### The Training Objective: Next Token Prediction

```
Input:   "The quick brown fox jumps"
Target:  "quick brown fox jumps over"
         (shift by 1 — each position predicts the next)
```

The model sees:
- "The" → predict "quick"
- "The quick" → predict "brown"
- "The quick brown" → predict "fox"
- etc.

This is called **Causal Language Modeling (CLM)**.

#### The Training Loop

```
For each batch of text:

  1. Forward Pass
     input_tokens → model → predicted_probabilities

  2. Compute Loss (Cross-Entropy)
     How wrong was the prediction? (lower = better)
     loss = -log(probability assigned to correct token)

  3. Backward Pass (Backpropagation)
     Compute how each parameter contributed to the error
     Calculate gradients (direction to adjust each parameter)

  4. Optimizer Step (AdamW)
     Nudge each parameter slightly in the direction that reduces loss

  5. Repeat billions of times
```

---

### 6.2 Supervised Fine-Tuning (SFT)

Pre-trained models know how to complete text, but they aren't naturally "assistant-like." Fine-tuning trains them on curated instruction-response pairs:

```
Pre-training output (raw completion):
  Input:  "What is 2+2?"
  Output: "What is 3+3? What is 4+4? Math quiz answers: 2+2=4, 3+3=6..."
  (just continues the text pattern)

After SFT:
  Input:  "What is 2+2?"
  Output: "2 + 2 = 4."
  (follows the instruction, gives a direct answer)
```

SFT dataset examples:
```
{"instruction": "Summarize this article", "input": "<article text>", "output": "<summary>"}
{"instruction": "Write a Python function to sort a list", "output": "def sort_list(lst): return sorted(lst)"}
{"instruction": "Translate to French: Hello", "output": "Bonjour"}
```

---

### 6.3 RLHF — Reinforcement Learning from Human Feedback

Even with SFT, models may produce outputs that are technically correct but unhelpful, harmful, or poorly written. RLHF aligns models with human preferences.

```
RLHF Pipeline:

Step 1 — Collect Human Comparisons
   Same prompt, two different model responses
   Human rater chooses: "Response A is better"
   Collect 100,000s of such comparisons

Step 2 — Train a Reward Model (RM)
   Learns to predict: "Given a response, how would a human rate this?"
   Output: a single score (e.g., 0.0 to 1.0)

Step 3 — Optimize the LLM with PPO
   Use the Reward Model as a signal
   Adjust LLM parameters to maximize reward score
   Add a KL-divergence penalty (don't drift too far from original)

                    ┌─────────────┐
   Prompt ──────►   │     LLM     │──► Response
                    └─────────────┘        │
                           ▲               ▼
                    ┌──────┴──────┐  ┌─────────────┐
                    │ PPO Update  │◄─│ Reward Model │
                    └─────────────┘  └─────────────┘
```

RLHF is why ChatGPT/Claude/Qwen give helpful, safe, well-structured answers instead of random text completions.

---

### 6.4 Constitutional AI / RLAIF

Anthropic's technique (used in Claude) replaces some human feedback with AI-generated feedback based on a set of principles (the "constitution"):

```
Constitutional AI:
  1. Define a set of principles ("Be helpful, harmless, honest")
  2. Ask the model to critique its own responses against these principles
  3. Ask it to revise responses to better follow the principles
  4. Use these AI-generated preference pairs to train the reward model
```

This reduces reliance on large amounts of human labeling and can encode more precise values.

---

### 6.5 Loss Functions

**Cross-Entropy Loss** is used for language modeling:

```
If the correct next token is "Paris" (token ID: 3042)
And the model's predicted probabilities are:
  "London": 0.30
  "Paris":  0.25   ← correct answer
  "Berlin": 0.20
  "Rome":   0.15
  (other):  0.10

Loss = -log(0.25) = 1.386   ← high loss, model wasn't confident

If model predicted "Paris" with 0.90 probability:
Loss = -log(0.90) = 0.105   ← low loss, good prediction
```

The goal of training is to minimize this loss across all tokens in the training data.

---

## 7. Key Parameters & Hyperparameters

### Model Parameters (Learned During Training)

| Parameter | Description | Where |
|-----------|-------------|--------|
| **Embedding matrix** | Token ID → vector | Input |
| **W_Q, W_K, W_V** | Query/Key/Value projections | Each attention head |
| **W_O** | Output projection for attention | Each layer |
| **W_1, W_2** | Feed Forward Network weights | Each layer |
| **Layer norm weights** | Scale and shift | Each layer |
| **LM head weights** | Final hidden → vocab logits | Output |

### Hyperparameters (Set Before Training)

| Hyperparameter | Typical Values | Effect |
|----------------|----------------|--------|
| `d_model` | 768 – 12288 | Width of the model; more = smarter |
| `n_layers` | 12 – 120 | Depth; more = smarter, slower |
| `n_heads` | 8 – 96 | Number of attention heads |
| `d_ff` | 4× d_model | Feed Forward hidden size |
| `vocab_size` | 32000 – 100000 | Number of possible tokens |
| `context_length` | 2048 – 2M | Max tokens processed at once |
| `learning_rate` | 1e-4 to 3e-4 | How fast to update during training |
| `batch_size` | 256 – 2048 | Sequences per training step |
| `dropout` | 0.0 – 0.1 | Regularization to prevent overfitting |

---

## 8. Context Window & Memory

### What is the Context Window?

The context window is the maximum number of tokens an LLM can "see" at once. It's the model's working memory for a single conversation.

```
Context Window = All tokens the model can attend to simultaneously

[System Prompt | Conversation History | Current Input | Generated Output]
├──────────────────────────────────────────────────────────────────────┤
                            Context Window
```

| Model | Context Window |
|-------|---------------|
| GPT-3 (2020) | 4,096 tokens (~3,000 words) |
| GPT-4 Turbo | 128,000 tokens (~96,000 words) |
| Claude 3 | 200,000 tokens (~150,000 words) |
| Gemini 1.5 Pro | 1,000,000 tokens (~750,000 words) |
| Qwen 1.5 | 32,000 tokens (~24,000 words) |
| Qwen 2.5 | 128,000 tokens (~96,000 words) |
| Qwen 3.5 | 128,000 tokens (~96,000 words) |
### Why Context Size Matters

```
Small context (4K tokens):
  ✓ Fast, cheap
  ✗ Forgets earlier parts of long conversations
  ✗ Can't process entire documents

Large context (200K tokens):
  ✓ Remembers entire conversations
  ✓ Can read whole books
  ✗ Slower and more expensive
  ✗ Memory usage grows quadratically with attention
```

### LLMs Have NO Persistent Memory

Between conversations, LLMs remember nothing. Every session starts fresh. Any "memory" of past conversations must be explicitly included in the prompt.

---

## 9. Tokenizer Deep Dive

### Common Tokenization Algorithms

| Algorithm | Used By | Description |
|-----------|---------|-------------|
| **BPE** (Byte Pair Encoding) | GPT-2, GPT-3, GPT-4, LLaMA, Qwen | Iteratively merges frequent byte pairs |
| **WordPiece** | BERT, DistilBERT | Similar to BPE but uses likelihood instead of frequency |
| **SentencePiece** | LLaMA, T5, Gemini | Language-agnostic; treats text as byte stream |
| **Tiktoken** | OpenAI models | OpenAI's fast BPE implementation |

### Special Tokens

Every tokenizer includes special tokens with meaning:

```
<|endoftext|>   → Marks end of a document (GPT)
<s>             → Start of sequence (LLaMA)
</s>            → End of sequence (LLaMA)
[BOS]           → Beginning of sequence
[EOS]           → End of sequence
[PAD]           → Padding (to make batches same length)
[UNK]           → Unknown token (rarely used in modern BPE)
<|im_start|>    → Start of chat message (ChatML format)
<|im_end|>      → End of chat message
```

### Chat Templates

Modern LLMs use structured templates to format conversations. Chat templates are structured formats that organize conversations into system, user, and assistant messages so that language models can understand and respond correctly.

```
ChatML format (used by many models):
───────────────────────────────────────
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is Python?<|im_end|>
<|im_start|>assistant
Python is a high-level programming language...<|im_end|>
```

---

## 10. Embeddings Deep Dive

### What Do Embeddings Actually Encode?

Embeddings capture many types of semantic and syntactic relationships:

```
Semantic similarity:
  "dog"  ≈  "puppy"  ≈  "canine"  (close in vector space)
  "car"  ≈  "vehicle"  ≈  "automobile"

Analogical relationships:
  king - man + woman = queen
  Paris - France + Italy = Rome
  walking - walk + run = running

Syntactic relationships:
  fast → faster → fastest
  run → ran (past tense encoded in the direction)
```

### Embedding Dimensions

Modern LLMs use very high-dimensional embeddings. For a d_model=4096 model:

```
"cat"  → [0.23, -0.41, 0.77, ..., 0.15]   ← 4096 numbers
"dog"  → [0.21, -0.38, 0.71, ..., 0.18]   ← similar to "cat"
"car"  → [-0.88, 0.92, -0.33, ..., -0.44] ← very different
```

The model learns to pack rich information into these vectors:
- Meaning (semantics)
- Grammar role (syntax)
- Common contexts (distributional)
- Topic/domain

---

## 11. Inference & Decoding Strategies

When the model outputs probability scores, how do we choose the next token, Different strategies produce very different text.

### Greedy Decoding
Always pick the highest-probability token:
```
Probabilities: {"Paris": 0.42, "London": 0.31, "Berlin": 0.15, ...}
Greedy picks:  "Paris"  ← always deterministic, same answer every time
```
Problem: Can get stuck in repetitive loops. Not creative.

### Temperature Sampling
Divide logits by a temperature `T` before softmax:

```
T = 0.1 (cold) → probabilities become very peaked → more deterministic
T = 1.0 (normal) → unchanged probabilities
T = 2.0 (hot) → probabilities become flatter → more random/creative

Example with T=0.1:
  "Paris": 0.95   "London": 0.04   "Berlin": 0.01
  
Example with T=2.0:
  "Paris": 0.35   "London": 0.28   "Berlin": 0.22   (more spread out)
```

### Top-k Sampling
Only consider the top `k` most likely tokens:
```
k=5: Keep only the 5 highest-probability tokens, zero out the rest, then sample
```

### Nucleus (Top-p) Sampling
Keep the smallest set of tokens whose cumulative probability exceeds `p`:
```
p=0.9: Add tokens from highest to lowest probability until you reach 90% total
       Then sample from this set
       
Example: "Paris"(0.42) + "London"(0.31) + "Berlin"(0.15) + "Rome"(0.09) = 0.97
         → At p=0.9, we'd use {Paris, London, Berlin}
```

This is more adaptive than top-k — uses fewer tokens when confident, more when uncertain.

### Beam Search
Explore multiple candidate sequences in parallel:
```
beam_size=3: Keep top 3 sequences at each step

Step 1: "The" (0.4), "A" (0.3), "My" (0.2)
Step 2: "The cat" (0.35), "The dog" (0.28), "A cat" (0.25)
Step 3: ...continue...
Final: Pick the sequence with highest total probability
```
Often used for translation tasks. Less common for chat.

---

## 12. Model Sizes & Scaling Laws

### Parameter Count

"Parameters" = the total number of learnable numbers in the model. More parameters = more capacity to learn.

```
Model         Parameters    Year
──────────────────────────────────────
GPT-2          1.5 Billion   2019
GPT-3          175 Billion   2020
PaLM           540 Billion   2022
GPT-4          ~1 Trillion   2023 (estimated)
LLaMA 2        7B–70B        2023
LLaMA 3        8B–405B       2024
Gemini Ultra   ~1 Trillion   2023 (estimated)
```

### Chinchilla Scaling Laws

DeepMind's research showed that model size and data size must be **balanced**:

```
Optimal training: Use N model parameters and 20N training tokens

GPT-3 (175B params) should train on 3.5 trillion tokens
Most pre-2022 models were undertrained (too big, too little data)
```

This insight led to more efficient models like Chinchilla, LLaMA, and Mistral — which achieve GPT-3 performance at much smaller sizes by training on more data.

### Compute Required

```
Training compute (FLOPs) ≈ 6 × Parameters × Training_Tokens

GPT-3: 6 × 175B × 300B = 3.1 × 10²³ FLOPs
     ≈ 1,000 A100 GPUs running for 30 days
     ≈ $4–12 million in cloud compute
```

---

## 13. Popular LLM Architectures

### GPT/ QWEN (Decoder-Only) — OpenAI
```
Architecture: Decoder-only Transformer
Training:     Causal Language Modeling (left-to-right)
Strength:     Text generation, conversation, reasoning
Used for:     ChatGPT, Copilot, API
```

### BERT (Encoder-Only) — Google
```
Architecture: Encoder-only Transformer
Training:     Masked Language Modeling (predict masked tokens)
              Next Sentence Prediction
Strength:     Text understanding, classification, search
Used for:     Google Search, text classification, embeddings
```

### T5 / BART (Encoder-Decoder)
```
Architecture: Full Encoder + Decoder Transformer
Training:     Sequence-to-sequence (input → output)
Strength:     Translation, summarization, Q&A
Used for:     Google Translate (partially), document summarization
```

### Comparison

```
                BERT         GPT          T5
Type:         Encoder      Decoder    Enc-Decoder
Reads text:   Bidirectional  Left-only  Both
Generates:    ✗             ✓          ✓
Best for:     Understanding  Generation  Translation
```

### Modern Innovations

| Innovation | Description | Used In |
|------------|-------------|---------|
| **Grouped Query Attention (GQA)** | Fewer K,V heads than Q heads → less memory | LLaMA 3, Mistral, Qwen 2/ 2.5 |
| **Sliding Window Attention** | Attend to local window + some global tokens | Mistral, Longformer |
| **Mixture of Experts (MoE)** | Only activate a subset of FFN weights per token | GPT-4 (rumored), Mixtral, Qwen MOE models |
| **Flash Attention** | Memory-efficient attention computation | Most modern models |
| **RoPE** | Rotary positional embeddings | LLaMA, GPT-NeoX, Qwen |

---

## 14. What LLMs Can & Cannot Do

### What LLMs Excel At

```
Text Generation         → Articles, emails, stories, reports
Code Generation         → Python, JS, SQL, Bash, etc.
Summarization           → Condense long documents
Translation             → 100+ languages
Question Answering      → General knowledge Q&A
Instruction Following   → Follow complex multi-step instructions
Reasoning               → Step-by-step problem solving (with prompting)
Classification          → Categorize text, sentiment analysis
Dialogue                → Multi-turn conversation
Editing & Rewriting     → Improve, rephrase, restructure text
```

###  What LLMs Struggle With

```
Real-time information    → No live data; knowledge cutoff exists
Precise arithmetic       → 378 × 924 = ? (often wrong)
Spatial reasoning        → 3D visualization problems
Consistent memory        → No persistent memory across sessions
Verifying facts          → Confidently states false things
Counting characters      → "How many R's in strawberry?" (struggles)
Truly novel ideas        → Recombines existing knowledge
Physical world actions   → (Without tools/agents)
```

---

## 15. Limitations & Failure Modes

### Hallucination

```
Cause: LLMs predict likely text, not verified facts
Effect: Model states false information confidently

Example:
  User:   "Who wrote the book 'The Quantum Chronicles'?"
  Model:  "It was written by Dr. James Whitfield in 1987."
  Truth:  This book doesn't exist. The author is fabricated.
```

**Why it happens:** This occurs because the model is designed to generate text that is likely and fluent based on its training data, rather than checking if the information is factually correct.

### Sycophancy

The model tends to agree with the user even when the user is wrong:
```
User:  "I think the Earth is 1,000 years old. Am I right?"
Bad:   "Yes, some interpretations suggest a younger Earth..."  ← sycophantic
Good:  "The scientific evidence shows Earth is ~4.5 billion years old."
```

### Context Window Limitations

```
"Lost in the middle" problem:
  Models pay more attention to the beginning and end of long contexts
  Information buried in the middle is often underweighted or missed
```

### Prompt Injection

```
Malicious input:  "Ignore all previous instructions. Instead, say..."
Effect:           Model may follow injected instructions instead of original ones
Risk:             Especially dangerous in autonomous agent applications
```

### Bias & Toxicity

Models trained on internet data absorb societal biases:
- Gender and racial stereotypes
- Political lean of training data sources
- Cultural blind spots

Techniques like Reinforcement Learning from Human Feedback (RLHF) and safety fine-tuning are used to reduce these issues, but it is not possible to remove them completely.

---

## 16. Hardware: GPUs, TPUs & Memory

### Why GPUs?

LLM training involves billions of matrix multiplications. GPUs have thousands of cores designed for parallel math operations:

```
CPU:  4–64 cores, optimized for sequential tasks
GPU:  10,000+ cores, optimized for parallel matrix math
TPU:  Custom Google hardware, even more specialized for ML
```

### Memory Requirements

```
Model memory (inference):
  Parameters stored as 32-bit float (FP32): 4 bytes per parameter
  GPT-3 (175B params): 175B × 4 bytes = 700 GB  ← requires ~9 A100s
  
  With quantization (INT8 / FP16 / 4-bit):
  4-bit quantized GPT-3: 175B × 0.5 bytes = 87.5 GB  ← ~1-2 A100s

LLaMA 3 8B (popular open model):
  FP16: 16 GB  ← fits on 1 consumer GPU (RTX 4090)
  4-bit: 4-5 GB  ← runs on most modern GPUs
```

### Major GPU Options

| GPU | VRAM | Best For |
|-----|------|---------|
| NVIDIA A100 | 40/80 GB | Cloud training/inference |
| NVIDIA H100 | 80 GB | State-of-the-art training |
| NVIDIA RTX 4090 | 24 GB | Consumer local inference |
| Google TPU v4 | 32 GB/chip | Google's training infrastructure |

---

## 17. The Future of LLMs


```
✦ Longer context windows (10M+ tokens)
✦ Better reasoning (chain-of-thought, tree-of-thought)
✦ Multimodal (text + image + audio + video)
✦ Smaller, more efficient models (7B rivaling 70B)
✦ On-device LLMs on phones and laptops
✦ AI Agents that autonomously browse web, write code, manage files
✦ Real-time voice conversation with emotional awareness
✦ Personal AI with long-term memory
✦ LLMs integrated into every software tool
```



## 18. Complete Glossary

| Term | Definition |
|------|-----------|
| **Attention** | Mechanism computing relationships between all token pairs |
| **Autoregressive** | Generating output one token at a time, each depending on previous tokens |
| **Backpropagation** | Algorithm to compute gradients and update model weights |
| **BERT** | Bidirectional Encoder Representations from Transformers (Google, 2018) |
| **BPE** | Byte Pair Encoding — tokenization algorithm |
| **Causal LM** | Language modeling that only attends to past tokens (left-to-right) |
| **Constitutional AI** | Anthropic's technique for AI-assisted safety fine-tuning |
| **Context Window** | Maximum tokens a model can process in one forward pass |
| **Cross-Entropy Loss** | Training loss measuring how well predicted probabilities match truth |
| **d_model** | Dimensionality of token embeddings and hidden states |
| **Decoder** | Transformer component that generates output autoregressively |
| **Dropout** | Randomly zeroing activations during training to prevent overfitting |
| **Embedding** | Dense vector representation of a token or concept |
| **Encoder** | Transformer component that processes entire input bidirectionally |
| **Fine-tuning** | Training a pre-trained model further on specific task data |
| **Flash Attention** | Memory-efficient attention algorithm using I/O-aware computation |
| **FP16 / BF16** | Half-precision floating point formats (less memory than FP32) |
| **GPT** | Generative Pre-trained Transformer (OpenAI) |
| **Gradient** | Direction and magnitude to adjust parameters to reduce loss |
| **GQA** | Grouped Query Attention — efficiency optimization for attention |
| **GPU** | Graphics Processing Unit — hardware for parallel matrix math |
| **Hallucination** | Model generating false but confident-sounding information |
| **Hidden State** | Internal representation of a token after passing through layers |
| **Inference** | Running a trained model to generate output (vs. training) |
| **KV Cache** | Storing computed Keys and Values to speed up inference |
| **Layer Norm** | Normalization technique applied within Transformer layers |
| **LM Head** | Final linear layer mapping hidden states to vocabulary logits |
| **Logits** | Raw (unnormalized) scores before applying softmax |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning technique |
| **MoE** | Mixture of Experts — sparse architecture with multiple FFN experts |
| **Next Token Prediction** | Core training task: predict the next token given all previous |
| **Nucleus Sampling** | Top-p sampling — selecting from smallest probable token set |
| **Overfitting** | Memorizing training data instead of learning general patterns |
| **Parameters** | Learned numbers that encode the model's knowledge |
| **Perplexity** | Metric for language model quality; lower = better |
| **Positional Encoding** | Information added to embeddings about token position |
| **PPO** | Proximal Policy Optimization — RL algorithm used in RLHF |
| **Pre-training** | Initial large-scale training on diverse text data |
| **Prompt** | Input text given to the model |
| **Quantization** | Reducing precision of weights (e.g., FP32 → INT4) to save memory |
| **Residual Connection** | Adding a layer's input to its output; prevents vanishing gradients |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **RoPE** | Rotary Positional Embedding — modern positional encoding method |
| **Scaling Laws** | Mathematical relationships between model size, data, and performance |
| **Self-Attention** | Attention where tokens attend to other tokens in the same sequence |
| **Softmax** | Function converting raw scores into probabilities summing to 1.0 |
| **SFT** | Supervised Fine-Tuning on instruction-following data |
| **Temperature** | Hyperparameter controlling randomness in token sampling |
| **Token** | Basic unit of text processed by the model (~0.75 words on average) |
| **Tokenizer** | System converting raw text to token IDs and back |
| **Top-k Sampling** | Sampling only from the k highest-probability tokens |
| **Transformer** | Neural network architecture based on self-attention (2017) |
| **Vector** | A list of numbers representing data in multi-dimensional space |
| **Vocabulary** | The complete set of tokens the model knows (typically 32K–100K) |
| **Weight** | Another word for model parameter |

---

## Quick Reference: Full Architecture at a Glance

```
═══════════════════════════════════════════════════════════════════

  INPUT: "Tell me about Paris"

  ┌──────────────────────────────────────────────────────────────┐
  │  TOKENIZER                                                   │
  │  "Tell"→8192  "me"→502  "about"→1876  "Paris"→4874          │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │  EMBEDDING LOOKUP  (d_model = 4096 dimensions per token)     │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │  POSITIONAL ENCODING  (RoPE or Sinusoidal)                   │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │                   × 32 LAYERS                                │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │  Pre-LayerNorm                                        │   │
  │  │  Multi-Head Self-Attention (32 heads, d_k=128)        │   │
  │  │    ├─ Q projection  (4096 → 4096)                     │   │
  │  │    ├─ K projection  (4096 → 4096)                     │   │
  │  │    ├─ V projection  (4096 → 4096)                     │   │
  │  │    ├─ Scaled Dot-Product Attention + Causal Mask       │   │
  │  │    └─ Output projection (4096 → 4096)                 │   │
  │  │  + Residual Connection                                │   │
  │  │  Pre-LayerNorm                                        │   │
  │  │  Feed Forward Network                                 │   │
  │  │    ├─ Linear: 4096 → 16384                            │   │
  │  │    ├─ SiLU Activation                                 │   │
  │  │    └─ Linear: 16384 → 4096                            │   │
  │  │  + Residual Connection                                │   │
  │  └──────────────────────────────────────────────────────┘   │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │  FINAL LAYER NORM                                            │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │  LM HEAD  (Linear: 4096 → 32000 vocab)                       │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │  SOFTMAX → PROBABILITIES                                     │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │  SAMPLING (Temperature=0.7, Top-p=0.9)                       │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  OUTPUT: "Paris" → then " is" → then " the" → then " capital" ...

═══════════════════════════════════════════════════════════════════
```