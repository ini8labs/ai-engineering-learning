# Week 1: Overview of Large Language Models



---

## Table of Contents

1. [What is AI / ML / Deep Learning?](#1-what-is-ai--ml--deep-learning)
   - [1.1 The Hierarchy](#11-the-hierarchy)
   - [1.2 Traditional Programming vs. the ML Paradigm](#12-traditional-programming-vs-the-ml-paradigm)
2. [Neural Networks from Scratch](#2-neural-networks-from-scratch)
   - [2.1 What is a Neuron?](#21-what-is-a-neuron)
   - [2.2 Mathematical Model of a Neuron](#22-mathematical-model-of-a-neuron)
   - [2.3 Activation Functions](#23-activation-functions)
   - [2.4 Forward Propagation](#24-forward-propagation-step-by-step)
   - [2.5 Loss Functions](#25-loss-functions)
   - [2.6 Backpropagation](#26-backpropagation)
   - [2.7 Learning Rate, Epochs, and Batch Size](#27-learning-rate-epochs-and-batch-size)
   - [2.8 Practical: Neural Network from Scratch (MNIST)](#28-practical-neural-network-from-scratch-mnist)
3. [Types of Learning](#3-types-of-learning)
   - [3.1 Supervised Learning](#31-supervised-learning)
   - [3.2 Self-Supervised Learning](#32-self-supervised-learning)
   - [3.3 Contrastive Learning](#33-contrastive-learning)
   - [3.4 Reinforcement Learning](#34-reinforcement-learning)
4. [Large Language Models (LLMs)](#4-large-language-models-llms)
   - [4.1 What Makes a Model "Large"?](#41-what-makes-a-model-large)
   - [4.2 Timeline of LLMs](#42-timeline-of-llms)
   - [4.3 How LLMs Work at a High Level](#43-how-llms-work-at-a-high-level)
   - [4.4 Emergent Abilities](#44-emergent-abilities)
5. [Scaling Laws](#5-scaling-laws)
   - [5.1 Chinchilla Scaling Laws](#51-chinchilla-scaling-laws)
   - [5.2 The Compute-Parameter-Data Relationship](#52-the-compute-parameter-data-relationship)
   - [5.3 Why Scaling Works (and Its Limits)](#53-why-scaling-works-and-its-limits)
   - [5.4 When to Use Small vs. Large Models](#54-when-to-use-small-vs-large-models)


---

## 1. What is AI / ML / Deep Learning?

### 1.1 The Hierarchy

Think of three concentric circles. The outermost is **Artificial Intelligence (AI)**, the middle is **Machine Learning (ML)**, and the innermost is **Deep Learning (DL)**.


---
```
┌─────────────────────────────────────────┐
│         Artificial Intelligence         │  ← Rule-based systems, search, planning
│   ┌─────────────────────────────────┐   │
│   │       Machine Learning          │   │  ← Learns from data: trees, SVM, regression
│   │   ┌─────────────────────────┐   │   │
│   │   │      Deep Learning      │   │   │  ← Deep neural networks with many layers
│   │   │  ┌───────────────────┐  │   │   │
│   │   │  │       LLMs        │  │   │   │  ← GPT, Claude, Llama
│   │   │  └───────────────────┘  │   │   │
│   │   └─────────────────────────┘   │   │
│   └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```


---



| Layer | Definition | Examples |
|-------|-----------|---------|
| **AI** | Any technique enabling a computer to mimic human-like intelligence | Chess engines, search algorithms, expert systems |
| **ML** | A subset of AI where systems learn from data rather than explicit rules | Decision trees, SVMs, neural networks |
| **DL** | A subset of ML using deep neural networks (many layers) to learn hierarchical representations | Image recognition, speech synthesis, LLMs |

> **Real-World Analogy — Teaching Someone to Cook:**
> - **AI (rule-based):** Hand them a detailed recipe with exact steps.
> - **ML:** Show them 1,000 photos of well-cooked vs. burnt dishes and let them figure out "good cooking."
> - **DL:** Give them millions of cooking videos; the network figures out everything from ingredient identification to plating, entirely by itself.

---

### 1.2 Traditional Programming vs. the ML Paradigm

One of the most important conceptual shifts in computer science:

| Aspect | Traditional Programming | Machine Learning |
|--------|------------------------|-----------------|
| **Input** | Data + Rules | Data + Expected Outputs |
| **Output** | Answers | Rules (learned model) |
| **Example** | `if email contains "free money" → spam` | Show 100k labeled emails; model learns patterns |
| **Maintenance** | Manually update rules as spammers evolve | Retrain on new data; model adapts automatically |
| **Scalability** | Rules become unmanageable in complex domains | Scales gracefully with more data |

**Key Insight:** The ML version can detect *new* spam patterns it has never explicitly seen, as long as they share statistical features with known spam. This is the power of learning from data.

---

## 2. Neural Networks from Scratch

### 2.1 What is a Neuron?

A **biological neuron** receives electrical signals through its dendrites, processes them in the cell body, and if the combined signal exceeds a threshold, fires an output through its axon.


```

  x₁ ──(w₁)──┐
  x₂ ──(w₂)──┤──► Σ (weighted sum + bias b) ──► f(z) ──► output y
  x₃ ──(w₃)──┘

  z = w₁x₁ + w₂x₂ + w₃x₃ + b
  y = f(z)   ← activation function

```


An **artificial neuron** mirrors this:

| Biological | Artificial | Role |
|-----------|-----------|------|
| Dendrites | Inputs (x₁, x₂, ..., xₙ) | Receive signals |
| Synaptic strength | Weights (w₁, w₂, ..., wₙ) | Scale the importance of each input |
| Cell body threshold | Bias (b) | Intrinsic excitability |
| Fire / don't fire | Activation function f | Decision to activate |

**Output formula:** `y = f(Σ(wᵢ · xᵢ) + b)`

> **Analogy:** Deciding whether to go to a party. Your inputs are: friends going (x₁=1), nice weather (x₂=1), homework due (x₃=1). Weights represent how much each factor matters to you: friends (w₁=0.8), weather (w₂=0.3), homework (w₃=−0.6). If the weighted sum exceeds your threshold, you go!

---

### 2.2 Mathematical Model of a Neuron

**Step 1 — Linear Transformation (Weighted Sum):**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  =  wᵀx + b
```

**Step 2 — Non-linear Activation:**
```
y = f(z)
```

---

### 2.3 Activation Functions

Activation functions introduce **non-linearity**. Without them, stacking layers is equivalent to a single linear transformation, no matter how deep the network. They are the key to a neural network's power.

| Function | Formula | Range | Use Case | Pros | Cons |
|---------|---------|-------|----------|------|------|
| **ReLU** | `max(0, x)` | [0, ∞) | Hidden layers | Fast, mitigates vanishing gradient | Dying ReLU problem |
| **Sigmoid** | `1 / (1 + e⁻ˣ)` | (0, 1) | Binary output | Outputs interpretable as probabilities | Vanishing gradients |
| **Tanh** | `(eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)` | (−1, 1) | RNN hidden states | Zero-centered | Vanishing at extremes |
| **Softmax** | `eˣⁱ / Σeˣʲ` | (0,1), Σ=1 | Multi-class output | Converts logits to a probability distribution | Expensive for large vocab |
| **GELU** | `x · Φ(x)` | (−∞, ∞) | Modern transformers (GPT, BERT) | Smooth, allows small negatives | Slightly more complex |


```

ReLU            Sigmoid          Tanh             Softmax
  |   /           |  ___         1|  ___          [0.66]
  |  /            | /            0|               [0.24]  ← sums to 1
  | /             |/            -1|___            [0.10]
──┼──────    ─────┼─────    ──────┼──────
  0               0               0
max(0,x)      1/(1+e⁻ˣ)    (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)   eˣᵢ/Σeˣʲ
Range:[0,∞)   Range:(0,1)   Range:(-1,1)         Range:(0,1), Σ=1
Hidden layers  Binary output  RNN states          Multi-class output

```

---

### 2.4 Forward Propagation (Step by Step)

Forward propagation passes input through the network layer by layer to produce a prediction.

**Example (2 inputs → 2 hidden neurons with ReLU → 1 output with Sigmoid):**

```
Input:   x = [0.5, 0.8]

Step 1 (Hidden Linear):   z = Wx + b  →  [0.54, 0.28]
Step 2 (Hidden ReLU):     a = ReLU(z) →  [0.54, 0.28]
Step 3 (Output Linear):   z = Wa + b  →  [0.718]
Step 4 (Output Sigmoid):  ŷ = σ(0.718) → 0.6723

Prediction: 0.6723 → Class 1 (positive)
```

---

### 2.5 Loss Functions

A **loss function** measures how far predictions are from true values. Training minimizes the loss.

| Loss Function | Formula | Used For |
|--------------|---------|---------|
| **MSE** | `(1/n) Σ(yᵢ − ŷᵢ)²` | Regression |
| **Binary Cross-Entropy** | `−(1/n) Σ[y·log(ŷ) + (1−y)·log(1−ŷ)]` | Binary classification |
| **Categorical Cross-Entropy** | `−Σ yc · log(ŷc)` | Multi-class classification, **LLM training** |

> **Key Insight:** Categorical Cross-Entropy is the loss function used to train LLMs. When predicting the next token, `y_true` is the correct token (one-hot encoded) and `y_pred` is the model's probability distribution over the entire vocabulary.

---

### 2.6 Backpropagation

Backpropagation computes how much each weight contributed to the error using the **chain rule of calculus**, then updates weights to reduce the loss.

**Chain Rule Intuition:** If changing oven temperature (x) changes how cooked the food is (y), which changes customer happiness (z):
```
dz/dx = dz/dy · dy/dx
```

**Process:**
1. Run forward pass → get prediction and cache intermediate values
2. Compute output gradient: `dL/dz_output = ŷ − y` (for sigmoid + BCE)
3. Propagate gradient backward layer by layer
4. Apply chain rule through each activation (e.g., ReLU gradient = 1 if z > 0, else 0)
5. Update all weights: `W_new = W_old − η · ∂L/∂W`


```

FORWARD PASS  →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
  Input x ───► Hidden h ───► Output ŷ ───► Loss L

BACKWARD PASS ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
  ∂L/∂W₁ ◄─── ∂L/∂h ◄────── ∂L/∂ŷ ◄───── ∂L/∂L = 1

  Weight Update:  W_new = W_old − η × ∂L/∂W
                              ↑
                        learning rate

```
---

### 2.7 Learning Rate, Epochs, and Batch Size

| Hyperparameter | Definition | Too Small | Too Large |
|---------------|-----------|-----------|-----------|
| **Learning Rate (η)** | Step size for weight updates | Trains very slowly, may get stuck | Overshoots minimum, diverges |
| **Epochs** | Complete passes through the training dataset | Underfitting | Overfitting |
| **Batch Size** | Examples processed before each weight update | Very noisy updates (SGD, batch=1) | Stable but slow, high memory (Batch GD) |

> **Practical values:** Learning rate 0.001–0.01 for simple networks; 1e-4 to 3e-4 for transformers. Mini-batch size of 32–512 is the practical sweet spot.

---

### 2.8 Practical: Neural Network from Scratch (MNIST)

**Architecture:** `784 inputs → 128 hidden (ReLU) → 64 hidden (ReLU) → 10 outputs (Softmax)`

**Key implementation steps:**
1. Load and normalize MNIST data (60,000 training images, 10,000 test)
2. One-hot encode labels
3. Initialize weights with **He initialization**: `W ~ N(0, sqrt(2/n_in))` — designed for ReLU networks
4. Forward propagation through all layers
5. Compute categorical cross-entropy loss
6. Backward propagation using chain rule
7. Update parameters with mini-batch gradient descent
8. Repeat for 20 epochs

**Expected results:**
```
Epoch  1/20 | Loss: 0.5123 | Test Accuracy: 89.42%
Epoch  5/20 | Loss: 0.1847 | Test Accuracy: 95.13%
Epoch 10/20 | Loss: 0.1102 | Test Accuracy: 96.51%
Epoch 20/20 | Loss: 0.0498 | Test Accuracy: 97.31%
```

~97% accuracy with only NumPy — no frameworks needed!

---

## 3. Types of Learning

```
┌──────────────────────┬───────────────────────────────────────────────────┐
│ Learning Type        │ How it Works                                      │
├──────────────────────┼───────────────────────────────────────────────────┤
│ Supervised           │ Labeled input→output pairs. Human labels needed.  │
│ Self-Supervised      │ Labels derived from data itself. No humans needed.│
│ Contrastive          │ Pull similar pairs close, push different ones away.│
│ Reinforcement        │ Agent takes actions, gets rewards, learns policy.  │
└──────────────────────┴───────────────────────────────────────────────────┘
```

### 3.1 Supervised Learning

The model learns from **labeled data** — input-output pairs where the correct answer is provided during training.

- **Classification:** Predict a discrete category (binary, multi-class, multi-label)
- **Regression:** Predict a continuous value (house prices, temperature)

---

### 3.2 Self-Supervised Learning

The model creates its **own labels from the data itself**. This is how modern LLMs learn — no human labeling required.

```
BERT — Masked Language Model (Bidirectional)
  "The [MASK] sat on the [MASK]"
       ↑                  ↑
     "cat"              "mat"
  Sees LEFT and RIGHT context → great for understanding

GPT — Next Token Prediction (Autoregressive / Causal)
  "The cat sat on" → predict "the"
  "The cat sat on the" → predict "mat"
  Sees only LEFT context → great for generation
```

| Approach | Model | How | Context |
|----------|-------|-----|---------|
| **Masked LM (MLM)** | BERT | Randomly mask 15% of tokens; predict them | **Bidirectional** — sees left AND right context |
| **Next Token Prediction** | GPT, Claude, Llama | Given tokens, predict the next one | **Unidirectional (causal)** — only sees previous tokens |

**Why next-token prediction works:**
- Infinite free training data (every text on the internet)
- Predicting the next word requires understanding grammar, facts, reasoning, common sense, and context — all at once

---

### 3.3 Contrastive Learning

Train the model to bring **similar examples close** and push **dissimilar examples apart** in embedding space.

- **SimCLR:** (Simple Framework for Contrastive Learning of Visual Representations) Augment images into positive pairs; push all other batch images away. No labels needed.
- **CLIP:** (Contrastive Language–Image Pretraining) Train on 400M (image, caption) pairs. Learn to match images with their text descriptions. Enables zero-shot classification: "Is this more similar to 'a cat' or 'a dog'?"

---

### 3.4 Reinforcement Learning

An **agent** learns by interacting with an **environment**, taking **actions**, and receiving **rewards**. Goal: maximize cumulative reward.

```
Step 1          Step 2           Step 3           Step 4
Pre-train  ──►  Human       ──►  Reward      ──►  PPO
LLM             Rankings        Model            Fine-tuning
(self-          (A > B)         (learns          (optimize
supervised)                      prefs)           reward)
                                                     │
                                                     ▼
                                               Aligned LLM 

```
**Connection to LLMs — RLHF (Reinforcement Learning from Human Feedback):**

| Step | What Happens |
|------|-------------|
| **1. Pre-train LLM** | Self-supervised on massive text corpus |
| **2. Human Ranking** | Human rankers rate which response is better |
| **3. Reward Model** | Trained to predict human preferences |
| **4. PPO Fine-tuning** | LLM fine-tuned to maximize reward model score |

This is what makes ChatGPT, Claude, and other models helpful and aligned with human preferences. DeepSeek R1 notably used RL more extensively for reasoning capabilities.

---

## 4. Large Language Models (LLMs)

### 4.1 What Makes a Model "Large"?

The "large" refers primarily to the number of **parameters** (weights and biases). More parameters generally means more capacity to learn complex patterns.

This is how an LLM generate text:
```

Prompt: "The cat"
       │
       ▼
  TOKENIZE ──► EMBED ──► TRANSFORMER LAYERS ──► SOFTMAX ──► SAMPLE ──► " sat"
  "The"→[464]   [0.2,      (Attention ×N)       [0.12,       ↑           │
  "cat"→[3797]   -0.5,...]                        0.05,     "sat"         │
                                                  0.18,...]               ▼
                                                                    Append & repeat
                                                                    until [EOS]
```

| Model | Parameters | Year | Notable Feature |
|-------|-----------|------|----------------|
| GPT-1 | 117M | 2018 | First GPT; proved pre-training works |
| BERT-Base | 110M | 2018 | Bidirectional; revolutionized NLP benchmarks |
| GPT-2 | 1.5B | 2019 | "Too dangerous to release" (eventually released) |
| GPT-3 | 175B | 2020 | In-context learning, few-shot prompting |
| PaLM | 540B | 2022 | Chain-of-thought reasoning |
| Llama 2 | 7B–70B | 2023 | Open-source; democratized LLMs |
| GPT-4 | ~1.8T (MoE, rumored) | 2023 | Multimodal; major quality leap |
| Llama 3.1 | 8B–405B | 2024 | Open-source; competitive with GPT-4 |
| DeepSeek V3 | 671B (MoE, 37B active) | 2024 | Efficient MoE; strong for cost |
| DeepSeek R1 | 671B (MoE) | 2025 | Reasoning model; extensive RL training |
| Claude 4 (Opus/Sonnet) | Undisclosed | 2025 | Advanced reasoning; agentic capabilities |
| GPT-5 | Undisclosed | 2025 | Unified reasoning model |

---

### 4.2 Timeline of LLMs

| Year | Milestone |
|------|-----------|
| 2017 | "Attention Is All You Need" introduces the Transformer |
| 2018 | GPT-1 (117M), BERT — pre-training revolution |
| 2019 | GPT-2 (1.5B) — surprisingly coherent text generation |
| 2020 | GPT-3 (175B) — in-context/few-shot learning |
| 2022 | ChatGPT goes viral; Chinchilla scaling laws published |
| 2023 | GPT-4, Claude 2, Llama 2 (open-source revolution) |
| 2024 | Claude 3.5 Sonnet, Llama 3.1, DeepSeek V3; focus on efficiency and agents |
| 2025 | DeepSeek R1, Claude 4, GPT-5; reasoning and agentic AI dominate |
| 2026 | Focus on efficient inference, multi-modal agents, production integration |

---

### 4.3 How LLMs Work at a High Level

At their core, LLMs are **next-token prediction machines**:

```
1. Tokenize  → "Hello" → [15496]
2. Embed     → [0.2, -0.5, ...]
3. Transform → Attention × N layers (pattern recognition)
4. Predict   → Probability distribution over vocabulary
5. Sample    → Pick next token
6. Repeat    → Append token, generate the next one
```

**Autoregressive generation example:**
```
"The cat" → "sat" → "on" → "the" → "mat" → [EOS]
```

The magic is in the Transformer layers. After training on trillions of tokens, these layers encode grammar, world knowledge, reasoning patterns, stylistic understanding, and much more.

---

### 4.4 Emergent Abilities

Abilities that appear **suddenly as models get larger**, without being explicitly trained for:

| Ability | When It Emerged | Why It Matters |
|---------|----------------|---------------|
| **In-context learning** | GPT-3 (175B) | Learn new tasks from prompt examples alone |
| **Chain-of-thought reasoning** | ~100B+ models | "Think step by step" dramatically improves math/logic |
| **Code generation** | GPT-3+ | Models trained on text also learned to write code |
| **Cross-lingual translation** | Large models | Translate between languages not explicitly trained on |
| **Theory of mind** | Debated | Signs of understanding that others have different beliefs |

> **The Emergence Debate (2024–2026):** Recent research questions whether emergent abilities are truly "sudden." Some argue that with the right evaluation metrics, improvements are gradual and predictable. The practical reality remains: larger models can do things smaller models simply cannot.

---

## 5. Scaling Laws

### 5.1 Chinchilla Scaling Laws

In 2022, DeepMind's Chinchilla paper changed how the industry trains LLMs.

**Chinchilla's Optimal Training Rule:**
```
D_optimal ≈ 20 × N

Where:  D = number of training tokens
        N = number of parameters
```

A 10B parameter model should be trained on ~200B tokens.

**Why this was revolutionary:** GPT-3 (175B params) was trained on only 300B tokens — massively undertrained. It should have seen ~3.5T tokens. DeepMind's Chinchilla (70B params, 1.4T tokens) matched GPT-3's performance with 4× fewer parameters.

---

### 5.2 The Compute-Parameter-Data Relationship

```
C ≈ 6 × N × D

Where:  C = compute in FLOPs
        N = number of parameters
        D = number of training tokens
```

The factor of 6 accounts for ~2 FLOPs (multiply + add) per parameter per token on the forward pass, and ~4 FLOPs on the backward pass.

| Model | Params | Tokens | Chinchilla Optimal? |
|-------|--------|--------|---------------------|
| GPT-2 | 1.5B | 40B | Under-trained |
| GPT-3 | 175B | 300B | Under-trained |
| Chinchilla | 70B | 1.4T | ✓ Optimal |
| Llama 2 7B | 7B | 2T | Over-trained ✓ |
| Llama 3.1 8B | 8B | 15T | Over-trained ✓ (1875× ratio!) |
| DeepSeek V3 | 671B | 14.8T | Over-trained ✓ |
---

This is the Scaling Triangle: 
```
Compute (C)
                  ▲
                 /|\
                / | \
               /  |  \
              /   |   \
  Params (N) ◄────────► Data (D)

  C ≈ 6 × N × D       (FLOPs for training)
  D_optimal ≈ 20 × N  (Chinchilla's rule)
```


> **The "Over-training" Trend (2024–2026):** Modern models train far beyond Chinchilla's recommendations because inference cost matters more than training cost. A smaller model trained on more data is much cheaper to run millions of times in production.

---

### 5.3 Why Scaling Works (and Its Limits)

**Why it works:** Neural networks are universal function approximators. Language modeling requires understanding grammar, facts, reasoning, and common sense — more parameters let the model store and compose more of these patterns.

**Current limitations and innovations:**

| Challenge | Response |
|-----------|---------|
| **Data wall** — running out of high-quality internet text | Training on synthetic data from other LLMs |
| **Diminishing returns from pre-training** | Inference-time compute (chain-of-thought reasoning) |
| **High inference cost of large models** | Mixture-of-Experts (MoE) — 671B params, only 37B active per token |
| **Raw capability vs. usefulness gap** | Post-training (RLHF, DPO) dramatically improves helpfulness |

---

### 5.4 When to Use Small vs. Large Models

| Consideration | Small Models (1B–8B) | Large Models (70B+) / API |
|--------------|---------------------|--------------------------|
| **Latency** | Fast (single GPU or CPU) | Slower (multi-GPU or API call) |
| **Cost** | Low (self-hosted) | Higher (GPU cluster or API pricing) |
| **Quality** | Good for specific tasks when fine-tuned | Better general reasoning |
| **Privacy** | Data stays on your servers | API: data goes to provider |
| **Use Cases** | Classification, extraction, edge deployment | Complex reasoning, code, agentic workflows |

> ** Rule:**
> 1. Start with the cheapest model (GPT-4o-mini / Claude Haiku)
> 2. Evaluate quality on YOUR specific task
> 3. Only upgrade to a larger model if quality is insufficient
> 4. Consider fine-tuning a small model before jumping to a large one
> 5. Use large models for evaluation/labeling to improve small models

---



### Summary

- **AI ⊃ ML ⊃ DL:** Deep learning is a subset of machine learning, which is a subset of artificial intelligence.
- **Neural networks** are composed of neurons that compute weighted sums, apply activation functions, and learn through backpropagation.
- **Self-supervised learning** (especially next-token prediction) is how modern LLMs learn from raw text without human labels.
- **LLMs are next-token prediction machines** that have encoded an enormous amount of knowledge and reasoning ability from training on internet text.
- **Scale matters but has limits**, leading to innovations in efficiency (MoE), reasoning (inference-time compute), and alignment (RLHF/DPO).

---




## Further Reading

This section goes — it explains what each paper argues, why the argument matters, and what you should specifically take away from it.

---

### 1. "Attention Is All You Need" — Vaswani et al. (2017)

**Paper:** [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (all at Google; equal contributors)

#### The Problem They Were Solving

Before 2017, the dominant approach to sequence tasks like machine translation used **Recurrent Neural Networks (RNNs)** and **LSTMs**. These had a fundamental architectural constraint: they processed tokens **one at a time, sequentially**. This caused several problems:

- **Slow training** — you couldn't parallelize across a sequence; token 5 had to wait for tokens 1–4 to finish
- **Vanishing gradients over long sequences** — information from early tokens degraded before reaching later layers
- **Long-range dependency failures** — the word "it" at position 50 might refer to a noun at position 3, but that connection was hard to maintain over 47 sequential steps



#### The Core Idea

The paper introduces the **Transformer** architecture, built entirely on **self-attention mechanisms**. Every token can directly "attend to" (compare itself against) every other token **simultaneously** — not sequentially.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```
- **Q (Queries):** What each token is looking for
- **K (Keys):** What each token has to offer
- **V (Values):** The actual information to aggregate
- **√dₖ scaling:** Prevents dot products from growing too large in high dimensions, which would push softmax into a flat gradient region

**Multi-Head Attention:** Instead of one attention operation, the model runs `h` parallel attention heads, each using different learned projection matrices. Each head can focus on a different type of relationship — one head might track syntactic dependencies, another semantic similarity. Their outputs are concatenated and projected back.

**Positional Encoding:** Since attention has no inherent notion of word order (unlike RNNs), the paper adds positional encodings to token embeddings using sine and cosine functions at varying frequencies. This lets the model distinguish "The dog bit the man" from "The man bit the dog."

**Architecture:** An encoder-decoder structure with 6 layers each. Every layer contains: (1) multi-head self-attention and (2) a position-wise feed-forward network. Residual connections and layer normalization wrap each sublayer for training stability. The decoder adds a third sublayer: cross-attention over the encoder output (so the decoder can "look at" the input sentence while generating the output).

The transformer architecture:
```
Input Tokens
     │
     ▼
[Embedding + Positional Encoding]
     │         ↑
     │    "Where is each token in the sequence?"
     ▼
┌──────────────────────────┐
│  Multi-Head Self-Attention│  ← Run attention H times in parallel,
│                          │    each learning different relationships
├──────────────────────────┤
│  Add & LayerNorm         │  ← Residual connection (helps gradients flow)
├──────────────────────────┤
│  Feed-Forward Network    │  ← Per-token transformation
│  (2 linear layers + ReLU)│
├──────────────────────────┤
│  Add & LayerNorm         │
└──────────────────────────┘
     │  (× N layers stacked)
     ▼
  Output Logits → Softmax → Token Probabilities

```

#### Key Results

Base Transformer (65M params) achieved state-of-the-art on English→German translation, beating all prior RNN/CNN models with less training time.

#### Why It Matters

-- Full parallelization → train efficiently on GPUs.

-- Every token can directly attend to every other token → no fading long-range dependencies.

-- Became the foundation for GPT, BERT, Claude, and virtually every modern LLM.

#### What to Take Away

- **Attention replaces recurrence.** Each token attends to all others simultaneously — no sequential bottleneck, no vanishing gradient across long distances.
- **Multi-head attention** lets the model simultaneously capture different types of linguistic relationships.
- **Positional encoding** injects word-order information into an otherwise order-agnostic architecture.
- **Parallelism is the killer feature.** Training that took weeks on RNNs now takes hours on GPUs.
- **Causal masking** in the decoder (masking future tokens in the attention matrix) is what makes GPT-style autoregressive generation possible — the model cannot "peek" at tokens it hasn't generated yet.
- Architectural details like 6 layers and 8 heads were tuned it.

---

### 2. "Training Compute-Optimal Large Language Models" (The Chinchilla Paper) — Hoffmann et al. (2022)

** Paper:** [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

** Authors:** Jordan Hoffmann and 21 co-authors at DeepMind

#### The Problem They Were Solving

By 2022, the field had converged on a simple assumption: **bigger models = better models**. GPT-3 (175B), Gopher (280B), and Megatron (530B) all followed this logic, each training on roughly 300 billion tokens regardless of model size. Nobody had rigorously asked: *Given a fixed compute budget, what is the optimal balance between model size and training data?*

The conventional wisdom (influenced by the earlier Kaplan scaling laws, see below) said: invest heavily in parameters, and data plays a secondary role.

#### The Core Idea

The DeepMind team ran a systematic empirical study, training **over 400 language models** ranging from 70M to over 16B parameters on datasets from 5B to 500B tokens. By varying both dimensions at controlled compute budgets, they mapped the loss landscape and fit a scaling law.

Their central finding:

> **For compute-optimal training, model size and number of training tokens should scale equally: for every doubling of model size, double the number of training tokens.**

This translates to approximately:
```
D_optimal ≈ 20 × N    (20 tokens per parameter)
```

To validate this, they trained **Chinchilla** — a 70B parameter model on 1.4 trillion tokens — using the same compute as Gopher (280B). Despite having less than one-quarter the parameters, Chinchilla uniformly outperformed Gopher, GPT-3, Jurassic-1, and Megatron-Turing NLG on nearly every benchmark. On the MMLU benchmark, Chinchilla hit 67.5% accuracy, more than 7% above Gopher's 60%.

#### Key Results

| Model | Params | Tokens | Chinchilla Verdict |
|-------|--------|--------|--------------------|
| GPT-3 | 175B | 300B | Needs ~3.5T tokens to be optimal — massively undertrained |
| Gopher | 280B | 300B | Similarly undertrained |
| Chinchilla | 70B | 1.4T | Optimal — outperforms all of the above |

Crucially: a smaller, well-trained model is also **much cheaper to run in production**, since inference cost scales with model size.

#### The Post-Chinchilla "Over-training" Trend

The industry quickly moved *beyond* Chinchilla's 20:1 prescription — not because it was wrong, but because it optimized for training efficiency given a fixed compute budget, not inference efficiency given millions of users. For a production model:

- **Llama 3.1 8B** was trained on 15T tokens (~1,875× the Chinchilla ratio)
- **DeepSeek V3** trained on ~14.8T tokens

The logic: training is a one-time cost. Inference runs billions of times. Spending more compute to train a smaller but better model yields massive long-term savings.

#### Why It Matters

The Chinchilla paper fundamentally changed how AI labs plan model training. It shifted the field from "make the model bigger" to "balance model size and data quality carefully." It directly influenced LLaMA, Mistral, and nearly every open-source model released after 2022.

#### What to Take Away

- **Most pre-2022 LLMs were undertrained, not undersized.** The bottleneck was data, not parameters.
- The **20:1 tokens-per-parameter ratio** is the Chinchilla-optimal baseline for training efficiency — but modern practice exceeds it for inference efficiency.
- **Smaller, well-trained models can match larger undertrained ones**, both in quality and in deployment cost.
- **Inference cost compounds.** A model served to 10M users runs inference billions of times — even small per-call savings become enormous at scale.
- **The paper also established** that smaller models are easier to fine-tune and deployable on smaller hardware, multiplying their practical advantages.
- The core finding that model size and data should scale equally was later broadly confirmed, though some specific coefficient estimates have been revised by replication studies.

---

### 3. "Scaling Laws for Neural Language Models" — Kaplan et al. (2020)

**Paper:** [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)

** Authors:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei

#### The Problem They Were Solving

In 2020, training large language models was expensive and deeply uncertain. There was no principled answer to: *If I double my compute budget, how much better will my model get? Should I scale model size, training time, or dataset size?* Practitioners made these decisions largely by intuition.

#### The Core Idea

The OpenAI team trained dozens of Transformer language models at carefully controlled scales and measured the cross-entropy loss at each scale. They discovered a remarkable regularity: **model performance follows precise power laws** with respect to each of the three key resources.

**The three power-law relationships:**

1. **Model size (N):** `Loss ∝ N^(−0.076)` — larger models consistently achieve lower loss
2. **Dataset size (D):** `Loss ∝ D^(−0.095)` — more training tokens predictably reduce loss
3. **Compute (C):** `Loss ∝ C^(−0.050)` — total training compute reduces loss, with trends spanning **over seven orders of magnitude**

These relationships held across different model architectures, datasets, and a vast range of scales — suggesting they reflect something fundamental about how neural networks learn from data, not just an artifact of specific design choices.

**Additional key findings:**

- **Architecture barely matters within a wide range.** Network width vs. depth has minimal effect on final loss — only total parameter count (non-embedding) is predictive.
- **Larger models are dramatically more sample-efficient.** A model 10× larger needs far fewer tokens to reach the same loss as a smaller model trained to convergence.
- **Early stopping is optimal.** The compute-optimal approach involves training very large models on modest data and stopping significantly before convergence — because the loss from more parameters falls faster than the loss from more data, at the scales studied.

**Important :** This last finding — that you should bias toward model size over data — was what Chinchilla later corrected. Kaplan et al. found this relationship at scales up to ~10B parameters; Chinchilla, studying a wider range with more controlled experiments, found that the optimal balance is actually equal scaling of both.

#### Why It Matters

This paper transformed LLM development from an art into a science. Before it, researchers had to guess. After it, they could **predict a model's loss before training it**, given knowledge of N, D, and C. It directly justified the investment in GPT-3, gave confidence that scaling to 175B parameters would deliver meaningful improvements, and set the trajectory for the entire modern LLM era.

The "scaling hypothesis" — that performance continues improving predictably with scale — became the central guiding principle for AI research from 2020 onward and justified billions of dollars in compute investment.

#### What to Take Away

- **Performance follows smooth, predictable power laws** with model size, data, and compute — no sudden cliffs or ceilings within the observed ranges.
- **You can estimate a model's quality before training it.** This makes resource planning scientific rather than speculative.
- **Larger models are more sample-efficient** — they extract more signal from each training token.
- **The paper's compute-allocation recommendation** (bias toward model size) was later revised by Chinchilla — read both together for the complete picture.
- The power-law trends spanned seven orders of magnitude, suggesting these are genuine statistical regularities in how neural networks learn from language, not coincidences.
- This paper influenced how the entire field thinks about investment in AI compute, and its framework remains the foundation even as specific coefficients are updated.

---

### 4. "GPT in 60 Lines of NumPy" — Jay Mody (Blog Tutorial)

** Link:** [jaykmody.com/blog/gpt-from-scratch](https://jaykmody.com/blog/gpt-from-scratch)


#### What It Is

A meticulously written blog post that implements a minimal but complete GPT (Generative Pre-trained Transformer) using approximately 60 lines of pure NumPy — no PyTorch, no TensorFlow, no abstraction layers. It loads real GPT-2 weights and runs actual inference, so you can verify every step against a known-good model.

#### What It Covers

The post walks through every component of a GPT-style decoder-only Transformer from first principles:

| Component | What It Does | Why It's Included |
|-----------|-------------|------------------|
| **Tokenization** | Converts raw text to integer token IDs using BPE | Entry point: how text enters the model |
| **Token embeddings** | Maps each token ID to a learned vector | Gives each token a "meaning vector" |
| **Positional embeddings** | Adds position-aware vectors to token embeddings | Tells the model where in the sequence each token sits |
| **Causal self-attention** | Each token attends to all *previous* tokens | The core reasoning mechanism; mask prevents future-peeking |
| **Multi-head attention** | Runs multiple attention operations in parallel | Captures different relationship types simultaneously |
| **Feed-forward layers** | Position-wise MLP applied after attention | Transforms attended representations into richer features |
| **Layer normalization** | Normalizes activations at each sublayer | Keeps training stable in deep networks |
| **Autoregressive generation** | Samples one token at a time, feeding it back as input | How GPT produces text |

#### Why It Matters

Reading papers gives you the theory. Implementing from scratch gives you the intuition. When you write each matrix multiplication yourself — without a library calling `model.forward()` — you're forced to confront every shape, every normalization step, and every architectural decision. After reading this post, the Transformer is no longer a black box.

#### What to Take Away

- The Transformer's forward pass is a sequence of matrix multiplications, element-wise operations, and softmax calls. There is no magic — only arithmetic.
- **Causal masking** is implemented by setting future attention weights to −∞ before the softmax, causing them to contribute zero to the output. This one line of code is what makes GPT autoregressive.
- **Pre-norm vs. post-norm** (applying layer norm before or after each sublayer) is a subtle but real design choice. GPT-2 uses pre-norm; original "Attention Is All You Need" used post-norm. Pre-norm tends to train more stably at scale.
- The full model is surprisingly compact once you strip away framework boilerplate. Understanding the 60-line version makes reading framework source code much easier.
- This is the ideal companion to "Attention Is All You Need" — the paper gives you the architecture, this post makes it executable.

---

## Practical Exercise

**1. Task: Implement the MNIST neural network from scratch and experiment with different learning rates (0.001, 0.01, 0.1, 1.0). What happens?**

---
See the implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\overview_of_llms\mnist_nn_lr_experiment.ipynb`

**Learning Rate Experiment on MNIST**

This notebook implements a neural network from scratch using NumPy and evaluates how different learning rates affect training performance on the MNIST dataset.

**Dataset**

The MNIST dataset contains handwritten digit images.

- 60,000 training images

- 10,000 test images

**Image size:** `28 × 28 pixels`

Each image is flattened into a 784-dimensional vector and normalized.

`28 × 28 image → 784 input features`

**Model Architecture**

A simple 2-layer neural network is implemented.

`Input Layer: 784 neurons
Hidden Layer: 64 neurons (ReLU)
Output Layer: 10 neurons (Softmax)`

**Flow:**

`784 → 64 → 10`
**Experiment**

The goal is to study how learning rate affects model training.

The following learning rates were tested:

`0.001
0.01
0.1
1.0`

For each learning rate:

- The model is trained for 10 epochs

- Loss is recorded

- Training curves are plotted

**Result**

The loss curves show how different learning rates impact training:

Very small learning rate (0.001) → slow learning

Moderate learning rate (0.01 / 0.1) → stable and faster convergence

Very large learning rate (1.0) → unstable training

**Conclusion**

The experiment demonstrates that learning rate is a critical hyperparameter in neural network training.

A moderate learning rate provides stable and efficient learning, while extremely small or large values can slow down or destabilize training.

---

**2. Task: Add a third hidden layer to the network. Does accuracy improve?**

---

See the implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\overview_of_llms\3_hidden_layer_nn.ipynb`

**Model Comparison**

In this experiment, two neural network architectures were trained on the MNIST dataset using NumPy.

**Model 1 (Shallow Network)**

Architecture: `784 → 64 → 10`

- 1 hidden layer

- Uses ReLU activation

**Model 2 (Deep Network)**

Architecture: `784 → 128 → 64 → 32 → 10`

- 3 hidden layers

- Uses ReLU activation

**Result**

The 3-hidden-layer model performs better than the 1-hidden-layer model because deeper networks can learn more complex feature representations from the data.

**Conclusion**

Increasing network depth improved the model’s ability to capture patterns in the MNIST digits, leading to better performance.

---
**3. Task: Replace ReLU with Sigmoid in the hidden layers. How does training speed change?**

---

See the implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\overview_of_llms\relu_sigmoid_hl_nn.ipynb`

**ReLU vs Sigmoid Experiment**

This notebook builds a neural network from scratch using NumPy and compares ReLU and Sigmoid activation functions on the MNIST dataset.

**Model Architecture**
`784 → 128 → 64 → 10`

The same architecture is trained twice:

- **Model 1:** Sigmoid activation

- **Model 2:** ReLU activation

**Result**

The ReLU model trains faster and performs better, while Sigmoid learns slower due to vanishing gradients.

**Conclusion** 

ReLU works better than Sigmoid for training deeper neural networks.

---

**4. Use the OpenAI or Anthropic API to compare responses from different model sizes on the same prompt.**

( Here I am using free LLM experiments in Google Colab with open-source models hosted on Hugging Face, which do not require an API key)

---
See the implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\overview_of_llms\compare_response_llm.ipynb`

**LLM Response Comparison**

This notebook compares responses from different Large Language Models using the OpenAI compatible API through Python.

**Objective**

The goal is to evaluate how different models respond to the same prompt and measure:

- Response time

- Word count

- Generated output

**Experiment**

Multiple models are queried with the same input prompt.

For each model the notebook records:

- Model name

- Response time

- Word count

- Generated text

The results are then stored and displayed in a dataframe for comparison.

**Conclusion**

This experiment helps analyze speed and response differences between LLMs, making it easier to compare model performance on the same task.

---
