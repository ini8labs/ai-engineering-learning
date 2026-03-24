# Week 4: Build Your Own GPT from Scratch


---

## Table of Contents

1. [What Are We Building?](#what-are-we-building)
2. [Causal Attention — How AI "Reads"](#causal-attention--how-ai-reads)
3. [Autoregressive Models — How AI "Writes"](#autoregressive-models--how-ai-writes)
4. [The Full GPT Architecture](#the-full-gpt-architecture)
5. [Training the Model](#training-the-model)
6. [Understanding Parameters](#understanding-parameters)
7. [Quick Cheatsheet](#quick-cheatsheet)
8. [Exercises](#exercises)

---

## What Are We Building?

We are building a **mini GPT** — a language model that can:
- Read text
- Learn patterns from that text
- Generate new text that sounds similar

Think of it like teaching a kid to write by having them read thousands of books.
After reading enough, they can write sentences that *sound* like the books they read!

```
Training Phase:
   "The cat sat on the mat"
   "The dog ran in the park"
   "The bird flew over the tree"
        ↓
   [Model Learns Patterns]
        ↓
Generation Phase:
    "The ___" → "cat" → "sat" → "on" → ...
```

---

## Causal Attention — How AI "Reads"

### The Big Idea

When a model reads a sentence like `"The cat sat on it"`, and it reaches the word **"it"** — it needs to figure out what "it" refers to. It should look back at "cat"!

This is called **self-attention** — each word looks at other words to understand context.

```
Sentence: "The  cat  sat  on  it"
                           ↑
              When reading "it"...
              
  "The"  ──── 0.12 ──→  "it"
  "cat"  ──── 0.65 ──→  "it"   ← strongest connection!
  "sat"  ──── 0.18 ──→  "it"
  "on"   ──── 0.05 ──→  "it"
```

### Why "Causal" Masking?

During **training**, the model sees the whole sentence at once.
But we don't want it to **cheat** by looking at future words when predicting the next word!

**Solution:** We use a mask (a block) to hide future words.

```
The Causal Mask — A Lower Triangular Matrix:

          The  cat  sat  on   the  mat
The    [  1    ✗    ✗    ✗    ✗    ✗  ]
cat    [  1    1    ✗    ✗    ✗    ✗  ]
sat    [  1    1    1    ✗    ✗    ✗  ]
on     [  1    1    1    1    ✗    ✗  ]
the    [  1    1    1    1    1    ✗  ]
mat    [  1    1    1    1    1    1  ]

  1 = Can look at this word 
  ✗ = BLOCKED — future word 
```

Reading this row by row:
- `"The"` can only see itself
- `"cat"` can see `"The"` and itself
- `"mat"` can see ALL previous words — it has the full context!

### How the Mask Works in Code

```
Step 1: Calculate attention scores (how related are two words?)
Step 2: Set future positions to -infinity  ← the MASK
Step 3: Apply softmax (convert scores to probabilities)
        -infinity → becomes 0% probability
Step 4: Use those probabilities to combine word meanings
```

>  **Simple analogy:** Imagine reading a mystery novel. You can remember everything you've already read, but you can't flip ahead to see the ending. That's causal masking!

### Encoder vs Decoder

| Feature | Encoder (e.g., BERT) | Decoder (e.g., GPT, Claude) |
|---|---|---|
| **Attention** | Sees ALL words | Only sees PAST words |
| **Best for** | Understanding text | Generating text |
| **Training task** | Fill in [MASK] tokens | Predict next word |
| **Example models** | BERT, RoBERTa | GPT-4, Llama, Claude |

---

## Autoregressive Models — How AI "Writes"

### Next Token Prediction

The core trick of GPT-style models is simple:

> **"Given everything I've read so far, what word comes next?"**

And this happens at **every single position** at the same time during training!

```
Input text:  "The  cat  sat  on  the  mat"
              ↓    ↓    ↓    ↓    ↓
Predictions: "cat" "sat" "on" "the" "mat"

So from 1 sentence with 6 words,
we get 5 training signals in ONE go! 
```

### Teacher Forcing (Training vs Real Use)

There's an important difference between **training** and **generating**:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING (Teacher Forcing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[BOS] → predict "The"   correct
[BOS] "The" → predict "dog"  wrong! (should be "cat")
[BOS] "The" "cat" → ...       ← Uses CORRECT "cat", NOT the wrong "dog"

Key idea: Always feed the CORRECT word, even if the model got it wrong.
This prevents errors from snowballing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERATING (Autoregressive)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[BOS] → "The"
[BOS] "The" → "cat"
[BOS] "The" "cat" → "sat"   ← Uses its OWN predictions
[BOS] "The" "cat" "sat" → "on"
...one token at a time, forever...
```

### Decoding Strategies — How to Pick the Next Word

Once the model gives us probabilities for the next word, how do we choose?

```
Vocabulary probabilities after "The cat":
  "sat"   ████████████████  45%
  "ran"   ████████          25%
  "ate"   ████              15%
  "slept" ███               10%
  "flew"  █                  5%
```

**Option 1: Greedy** — Always pick the highest probability word
```
  → Always picks "sat"
   Fast and consistent
   Boring and repetitive
```

**Option 2: Top-K Sampling** — Pick randomly from the top K words
```
  K=3: Only consider "sat", "ran", "ate"
  → Randomly pick one of these 3
   More variety
   Fixed cutoff can be too loose or too tight
```

**Option 3: Top-P (Nucleus) Sampling** — Pick from words until probabilities add up to P%
```
  P=0.85:
    "sat"   45%  → cumulative: 45%  
    "ran"   25%  → cumulative: 70%  
    "ate"   15%  → cumulative: 85%   stop here!
    "slept" 10%  → excluded 
    "flew"   5%  → excluded 

   Adaptive — works well in practice
   Most commonly used in ChatGPT, Claude, etc.
```

**Temperature** controls how "random" or "confident" the model is:
```
Low Temperature (0.2) → Model is very confident, safe choices
  "sat" 88%, "ran" 8%, "ate" 3%...

High Temperature (2.0) → Model is adventurous, more random
  "sat" 28%, "ran" 24%, "ate" 20%, "slept" 16%, "flew" 12%...
```

---

## The Full GPT Architecture

### Bird's Eye View

```
Input Text: "Hello world"
      ↓
  [Tokenizer]  →  [84, 101, 108, 108, 111, ...]
      ↓
  [Token Embedding]   +   [Position Embedding]
      ↓
  ┌─────────────────────────────┐
  │     Transformer Block 1     │
  │  LayerNorm → Attention → +  │
  │  LayerNorm → FFN → +        │
  └─────────────────────────────┘
      ↓
  ┌─────────────────────────────┐
  │     Transformer Block 2     │
  └─────────────────────────────┘
      ↓
     ... (N blocks total) ...
      ↓
  [Final LayerNorm]
      ↓
  [Output Projection → Vocabulary]
      ↓
  Probabilities for next word 
```

### Inside One Transformer Block

```
Input x
  │
  ├─────────────────────────────────────┐
  │                                     │
  ▼                                     │
[LayerNorm]                             │  (normalize the values)
  │                                     │
  ▼                                     │
[Multi-Head Causal Self-Attention]      │
  │                                     │
  └─── + ──────────────────────────────┘  ← Residual (add input back!)
         │
         ├────────────────────────────────┐
         │                                │
         ▼                                │
     [LayerNorm]                          │
         │                                │
         ▼                                │
     [Feed-Forward Network]               │
     Linear → GELU → Linear              │
         │                                │
         └─── + ──────────────────────────┘  ← Residual again!
               │
               ▼
           Output x'
```

>  **What's a Residual Connection?** The `+` means we ADD the original input back to the output. This creates a "highway" for gradients during training and helps the model learn much faster. Without it, very deep networks are nearly impossible to train!

### Multi-Head Attention — Step by Step

```
Input: (batch=1, sequence=7 words, dimensions=384)

Step 1: Project to Q, K, V
  Q = input × W_Q   (What am I looking for?)
  K = input × W_K   (What do I contain?)
  V = input × W_V   (What will I pass along?)

Step 2: Split into multiple "heads"
  384 dimensions → 6 heads × 64 dimensions each
  (Each head learns different types of relationships!)

Step 3: Compute attention scores
  scores = Q × K^T / √64

Step 4: Apply causal mask
  (Set future positions to -∞)

Step 5: Softmax → attention weights

Step 6: Weighted sum of Values
  output = weights × V

Step 7: Merge all heads back together
  6 heads × 64 dims → 384 dims

Step 8: Final projection
  384 → 384
```

### Feed-Forward Network

This is simpler — just two linear layers with an activation in between:

```
Input (384 dims)
  ↓
Linear layer: 384 → 1536 dims  (expand 4x)
  ↓
GELU activation  (like ReLU but smoother)
  ↓
Linear layer: 1536 → 384 dims  (compress back)
  ↓
Output (384 dims)
```

>  **Why expand then compress?** This "bottleneck" lets the model learn richer, non-linear transformations. The expansion layer finds features; the compression layer selects the important ones.

---

## Training the Model

### The Loss Function

We use **Cross-Entropy Loss** — it measures how wrong the model's prediction was.

```
Example:
  Model predicts next word probabilities:
    "sat"  → 30%
    "ran"  → 45%  ← Model thinks this is most likely
    "ate"  → 15%
    ...

  But correct answer is "sat"

  Loss = -log(0.30) = 1.20   ← Higher = more wrong!

  If model was perfect (100% on "sat"):
  Loss = -log(1.00) = 0.00   ← Perfect!
```

**Perplexity** = e^(loss) — a more intuitive version:
- Perplexity of 10 means "model is as confused as guessing from 10 options"
- GPT-4 achieves perplexity ~5-10 on standard tests
- A random model with 50,000 vocab = perplexity ~50,000 

### The Training Loop

```
For each step (repeat thousands of times):

  1.  Get a batch of text sequences
  
  2.  Forward Pass
        Input text → Model → Predicted probabilities
  
  3.  Calculate Loss
        Compare predictions to actual next words
  
  4.  Backward Pass (Backpropagation)
        Calculate how to adjust every weight to reduce loss
  
  5.  Gradient Clipping
        If gradients are too large, scale them down
        (Prevents "exploding gradients" that destroy training)
  
  6.  Optimizer Step (AdamW)
        Update all weights using gradients
  
  7.  Log progress every 100 steps
```

### The AdamW Optimizer

AdamW is the standard optimizer for training language models. It's an improved version of classic gradient descent.

```
Regular Gradient Descent:
  weight = weight - learning_rate × gradient
  (Very slow to converge, same LR for all weights)

Adam:
  Tracks momentum (moving average of gradients)
  Tracks adaptive rates (bigger steps for rarely-updated weights)

AdamW (Adam + Weight Decay):
  Also applies a small "shrink" to all weights each step
  This acts as regularization — prevents overfitting

Key hyperparameters:
  lr = 3e-4           (peak learning rate)
  beta1 = 0.9         (momentum decay)
  beta2 = 0.95        (adaptive rate decay)
  weight_decay = 0.1  (regularization strength)
```

 **Important:** Don't apply weight decay to biases or LayerNorm parameters — only to weight matrices!

### Learning Rate Schedule

Don't use a fixed learning rate! The best approach is:

```
Learning Rate over Training:

  ▲
  │         ╭──────╮
  │        /        ╲
  │       /          ╲
  │      /            ╲
  │     /              ╲___________
  │    /                           ╲___
  │___/                                ╲___
  └──────────────────────────────────────→
     ^warmup^         ^cosine decay^      end
     (100 steps)      (rest of training)

Phase 1 - Warmup: LR slowly increases to peak
  Why? Weights are random at start — large LR causes chaos!

Phase 2 - Cosine Decay: LR slowly decreases
  Why? Fine-tune later in training with smaller, careful steps
```

### Gradient Clipping

```
Without clipping:
  One bad batch → giant gradient → weights fly off to infinity 
  Training crashes!

With clipping (max_norm = 1.0):
  If total gradient magnitude > 1.0:
    Scale ALL gradients down proportionally
  Result: smooth, stable training 

In code (applied after loss.backward()):
  grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### Mixed Precision Training (BF16)

```
Standard (FP32): 4 bytes per number
  Range: ±3.4 × 10^38
  Precision: very high

BF16: 2 bytes per number (HALF the memory)
  Range: same as FP32  (unlike FP16)
  Precision: slightly lower (acceptable)

Memory savings for a 7 Billion parameter model:
  FP32 training: ~112 GB GPU memory needed
  BF16 training: ~70 GB GPU memory needed
  
Speed: BF16 is roughly 2× faster on modern GPUs!

Recommended: Use BF16 for training in 2024+
```

---

## Understanding Parameters

### Where Do All the Parameters Live?

```
For a model with vocab=50,000, d_model=768, n_layers=12:

┌──────────────────────────────────────────────────────┐
│  Token Embedding    50,000 × 768     = 38.4M  (31%)  │
│  Position Embedding  1,024 × 768     =  0.8M  ( 1%)  │
├──────────────────────────────────────────────────────┤
│  × 12 Transformer Blocks:                             │
│    Attention Q,K,V  3 × 768 × 768   = 1.77M          │
│    Attention Out        768 × 768   = 0.59M          │
│    FFN Up           768 × 3072      = 2.36M          │
│    FFN Down         3072 × 768      = 2.36M          │
│    LayerNorms       4 × 768         = 0.003M         │
│    ────────────────────────────────────────          │
│    Total per layer                  ≈ 7.08M          │
│    × 12 layers                     = 85M   (68%)     │
├──────────────────────────────────────────────────────┤
│  Output Projection   (tied w/ embedding) = 0         │
│  Final LayerNorm                    =  1.5K          │
└──────────────────────────────────────────────────────┘
  TOTAL                               ≈ 124M params  
  (This is GPT-2 Small!)
```

### The Rule of Thumb

```
Total Parameters ≈ 12 × n_layers × d_model²

Examples:
  Mini-GPT:  12 × 6 × 384²   ≈  10.6M
  GPT-2:     12 × 12 × 768²  ≈  84.9M  (actual: 124M with embeddings)
  GPT-3:     12 × 96 × 12288² ≈ 174B   (actual: 175B )
```

### Common Model Sizes

| Model | d_model | n_layers | n_heads | Params |
|---|---|---|---|---|
| Our Mini-GPT | 384 | 6 | 6 | ~10M |
| GPT-2 Small | 768 | 12 | 12 | 124M |
| GPT-2 Large | 1280 | 36 | 20 | 774M |
| GPT-3 | 12288 | 96 | 96 | 175B |
| Llama 3.1 8B | 4096 | 32 | 32 | 8B |

---

## Quick Cheatsheet

```
┌─────────────────────────────────────────────────────────────┐
│                     GPT QUICK REFERENCE                      │
├─────────────────────┬───────────────────────────────────────┤
│ Causal Mask         │ Lower triangular matrix of 1s and 0s  │
│                     │ Future positions → set to -∞           │
├─────────────────────┼───────────────────────────────────────┤
│ Teacher Forcing     │ Feed correct tokens during training    │
│                     │ Use own predictions during inference   │
├─────────────────────┼───────────────────────────────────────┤
│ Loss Function       │ Cross-entropy on next token           │
│                     │ Perplexity = e^loss                   │
├─────────────────────┼───────────────────────────────────────┤
│ Optimizer           │ AdamW (lr=3e-4, β1=0.9, β2=0.95)     │
├─────────────────────┼───────────────────────────────────────┤
│ LR Schedule         │ Linear warmup → Cosine decay          │
├─────────────────────┼───────────────────────────────────────┤
│ Gradient Clipping   │ max_norm = 1.0                        │
├─────────────────────┼───────────────────────────────────────┤
│ Precision           │ BF16 (2x faster, same quality)        │
├─────────────────────┼───────────────────────────────────────┤
│ Decoding            │ Top-p (0.9-0.95) + Temperature (0.8)  │
├─────────────────────┼───────────────────────────────────────┤
│ Param estimate      │ ≈ 12 × n_layers × d_model²            │
└─────────────────────┴───────────────────────────────────────┘
```


---

## Exercises

---

**Implement SwiGLU activation in the FFN. Compare training curves with GELU.**

---

See implementation in `E:\ini8_labs\ai-engineering-learning\casual_attention_coding_transformer\swiglu_gelu_comparisson.ipynb`

**SwiGLU vs GELU in Mini-GPT**



**Overview**

This project compares two activation functions in a Transformer (Mini-GPT):

* **GELU** (standard)
* **SwiGLU** (modern, gated)

We train both models on the Tiny Shakespeare dataset and compare their training loss.


**What the Code Does**

1. Loads text data (Shakespeare)
2. Converts text into tokens (numbers)
3. Builds a small GPT model
4. Uses:

   * GELU in one model
   * SwiGLU in another
5. Trains both models
6. Plots training loss graph


**Key Idea**

### GELU

* Simple activation function
* Learns faster in the beginning

**SwiGLU**

* Uses a **gating mechanism**
* Learns slower at first
* Performs better later


**Result (From Graph)**

* Both models learn properly 
* GELU improves faster early
* SwiGLU catches up later
* SwiGLU often gets **slightly lower final loss**


**Conclusion**

* GELU → good for fast learning
* SwiGLU → better for final performance
* SwiGLU is used in modern large models

---

**Modify the model to use RMSNorm instead of LayerNorm. Does training stability change?**

---

See implementation in `E:\ini8_labs\ai-engineering-learning\casual_attention_coding_transformer\comparison_layernorm_rmsnorm.ipynb`


---

## Further Reading

**1. "Language Models are Few-Shot Learners" (Brown et al., 2020) — GPT-3 paper**

---


**Overview**

This paper introduces **GPT-3**, a very large language model that can perform many tasks **without task-specific training**.
Instead of fine-tuning, it learns from **examples given in the prompt**.


**Why this paper is important**

Before GPT-3:

* Models needed **separate training for each task**
* Required **labeled data**
* Not flexible

GPT-3 changed this by showing:

  One model can do many tasks using just prompts


**Key Idea: Few-Shot Learning**

GPT-3 can learn tasks from examples inside the input.

**Example:**

```
English → French
dog → chien
cat → chat
house →
```

Output: *maison*

* No training required
* Learns from context

**Types of Learning**

* **Zero-shot** → No examples
* **One-shot** → One example
* **Few-shot** → Few examples


**How GPT-3 Works**

* Based on **Transformer architecture**
* Trained on **large internet text**
* Predicts the **next word in a sequence**
* Uses patterns learned during training


**Key Insight**

> Bigger models + more data = better performance

This idea is called **scaling**.


**What GPT-3 Can Do**

* Translation
* Question answering
* Summarization
* Text generation
* Basic coding

All using the same model


**Advantages**

* No need for fine-tuning
* Works for many tasks
* Needs very little data (few examples)
* Easy to use (just write prompts)
* Strong generalization ability


**Disadvantages**

* Very expensive to train
* Can give wrong answers (hallucination)
* Sensitive to prompt wording
* No real understanding (just pattern prediction)
* Can contain bias from training data


**Important Note**

GPT-3 does **not actually learn during use**
It only uses patterns learned during training to predict outputs

**Why This Paper Changed AI**

Before:

* Task-specific models

After:

* General-purpose language models

Led to modern AI systems like chatbots and assistants


**Simple Analogy**

* Old models → need training for every task
* GPT-3 → learns from examples instantly


**Final Takeaway**

> A very large language model can perform many tasks using just a few examples, without retraining.



---

**2. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2019) — T5 paper**

---



**What is this paper about**

This paper introduces **T5 (Text-to-Text Transfer Transformer)**.

Main idea:
Convert **every NLP task into a text-to-text problem**

* Input = text
* Output = text

One model can do **all tasks**

**Why this paper came**

Before T5:

* Different models for different tasks
* Complex pipelines
* Hard to compare methods

T5 solves this by:

* Using **one unified framework**
* Making everything simple and consistent


**Core Idea**

All tasks → **Text → Text format**

**Examples:**

* Sentiment:

  ```
  "sentiment: I love this movie" → "positive"
  ```

* Translation:

  ```
  "translate English to German: Hello" → "Hallo"
  ```

* Question Answering:

  ```
  "question: What is AI? context: ..." → "Artificial Intelligence"
  ```



**Model Architecture**

* Based on **Transformer**
* Uses:

  * Encoder → understands input
  * Decoder → generates output

Works for both understanding + generation tasks



**Training Method**

**Pretraining:**

* Dataset: Large cleaned web data (C4)

**Objective:**

**Fill in the blanks (Denoising)**

Example:

```
Input:  "I love <mask> learning"
Output: "deep"
```

 Model learns context better



**Key Findings**

* Encoder–Decoder works better than other setups
* Clean and large data is very important
* Denoising objective gives best results
* Bigger models perform better


**Advantages**

* One model for all NLP tasks
* Simple and flexible
* Strong performance
* Easy multi-task learning
* Good for transfer learning


**Disadvantages**

* Requires high compute power
* Slow during text generation
* Hard to train from scratch
* Not always best for specialized tasks


**Why T5 is Important**

* Simplified NLP pipeline
* Introduced unified framework
* Influenced modern LLMs
* Helped in prompt-based learning


**Key Takeaways**

* Convert everything to **text-to-text**
* Use encoder–decoder transformer
* Train using fill-in-the-blanks
* Data quality matters a lot
* Bigger models = better results

**One Line Summary**

T5 =
**One model that solves all NLP tasks using text input and text output**

---

**3. nanoGPT by Andrej Karpathy — Minimal GPT implementation**

---

**What is nanoGPT**

**nanoGPT** is a small and simple version of GPT (like ChatGPT) made for learning.

It is created by Andrej Karpathy to help people understand how GPT works inside.


**Main Idea**

GPT models work by:

> Predicting the **next word (or character)** in a sentence.

Example:

```
Input:  "I love"
Output: "coding" / "AI"
```

This is called **next token prediction**.


**Why nanoGPT was created**

Before nanoGPT:

* GPT models were very large
* Code was complex
* Hard for beginners to understand

nanoGPT solves this by:

* Keeping code **small and clean**
* Showing only **important parts**
* Making learning **easy**


**What nanoGPT contains**

**1. Model**

* Transformer-based GPT
* Includes:

  * Attention
  * Feedforward layers


**2. Training**

* Reads text data
* Learns patterns
* Updates model weights


**3. Text Generation**

* Generates text like:

```
ROMEO: What light through yonder window breaks?
```


**How it works**

```
Text → Tokens → Train Model → Generate Text
```


**Key Features**

* Very small code (easy to read)
* Runs on normal laptop/GPU
* Easy to modify
* Good for experiments



**Advantages**

* Easy to understand
* Great for beginners
* Full control of model
* Helps learn transformers deeply
* Fast experimentation


**Disadvantages**

* Not for real-world production
* Cannot train very large models
* Limited features
* Less optimized

**Important Concepts You Learn**

**1. Next Token Prediction**

Model learns to guess next word


**2. Attention Mechanism**

Helps model understand relationships between words


**3. Training Loop**

* Forward pass
* Loss calculation
* Backpropagation


**4. Tokenization**

Converting text into numbers


**5. Text Generation**

Using:

* Temperature
* Top-k / Top-p


**Important Lessons**

* Data quality matters more than model size
* Training process is very important
* Small changes can affect results a lot
* Bigger models need more compute



**When to Use nanoGPT**

Use it if:

* You want to learn GPT
* You want to experiment
* You want to understand transformers

Do NOT use it if:

* You want to build production apps
* You need large-scale models


**Final Summary**

nanoGPT is:

* A **learning tool**
* A **minimal GPT implementation**
* A **playground for experiments**

---

**4. "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)**

---



**What is LLaMA 2**

LLaMA 2 is a **large language model (LLM)** like ChatGPT.

It can:

* Answer questions
* Generate text
* Help in coding
* Chat like a human

It comes in different sizes:

* 7B (small)
* 13B (medium)
* 70B (large)


**Why was this paper created**

Before LLaMA 2:

* Powerful models (like GPT-4) were **closed**
* People could not use or modify them

Meta wanted to:

* Make AI **open and accessible**
* Give developers **control**
* Enable research and innovation



**Types of Models**

**1. Base Model**

* Just predicts next word
* Not good for chatting

**2. Chat Model (LLaMA 2-Chat)**

* Trained for conversation
* More helpful and safe


**How it works**

**Step 1: Pretraining**

* Learns from large internet text

**Step 2: Supervised Fine-Tuning (SFT)**

* Humans give good answers
* Model learns how to respond properly

**Step 3: RLHF (Human Feedback)**

* Humans rank answers
* Model improves based on feedback


**Key Ideas**

* Good **data quality** is very important
* Bigger models are better, but expensive
* Chat models need **human feedback**
* Safety is still a challenge


**Improvements over LLaMA 1**

* Better training data
* More stable performance
* Longer context (remembers more text)
* Faster attention (GQA)
* Better chat ability


**Advantages**

* Open and usable
* Strong performance
* Can be customized (fine-tuning)
* Cheaper than API-based models
* Good for research and projects


**Disadvantages**

* Not as powerful as top closed models
* Needs high GPU for large versions
* Can give wrong answers (hallucination)
* Safety not perfect
* Requires tuning for best results


**Is it fully open**

 Not fully

*  Model weights are available
*  Training data not fully shared

 So it is **partially open**



**Real-World Uses**

* Chatbots
* Coding assistants
* AI in DevOps
* Research experiments
* Private AI systems


**Key Takeaways**

* LLaMA 2 made AI **more accessible**
* Open models can compete with closed ones
* Human feedback (RLHF) is very important
* Data quality matters more than size
* It started the **open LLM movement**


**One-Line Summary**

 **LLaMA 2 = Open-source ChatGPT-like model that you can use and customize**



---

**5. "The Llama 3 Herd of Models" (Dubey et al., 2024)**

---

**What is this paper**

Llama 3 is a family of powerful AI language models created by Meta.

Means:
- Not one model
- But multiple models of different sizes

These models can:
- Chat 
- Write code 
- Solve problems 
- Understand long text 


**Why did Llama 3 come**

Before this:
- Powerful models (like GPT-4) existed
- But they were mostly closed (not accessible)

Meta wanted to:
- Build powerful models
- Make them more open and usable
- Compete with top AI systems


There are multiple models:

| Model | Use |
|------|-----|
| 8B | Fast, cheap tasks |
| 70B | Balanced tasks |
| 405B | Very powerful tasks |

One model cannot fit all needs  
So they created a “team of models”


**Key Ideas**

**1. Bigger + Better Training**
- Trained on huge data (~15T tokens)
- More data = better performance


**2. Same Architecture**
- Uses Transformer (not new architecture)

Improvement comes from:
- Data
- Training process


**3. Strong Capabilities**
- Reasoning
- Coding
- Multi-language
- Long context understanding


**4. Alignment**
- Fine-tuned to be helpful and safe
- Raw model → smart but risky
- Aligned model → useful


**Advantages**

* Open-weight (can be used by developers)  
* Very strong performance  
* Different sizes for different needs  
* One model can do many tasks  
* Long context support  



**Disadvantages**

* Large models are very expensive  
* Not fully open (license restrictions)  
* Can hallucinate (give wrong answers)  
* Safety risks (can be misused)  


**Important Learnings**

**1. Scaling works**
Bigger model + more data = better results


**2. Data is most important**
Good data > fancy architecture


**3. Training pipeline matters**
- Data cleaning
- Deduplication
- Smart training order


**4. Alignment is key**

Makes model safe and usable


**5. System thinking**

Not one model → multiple models (herd)



**Final Summary**

Llama 3 shows that:

- You don’t need new architecture  
- You need better data and training  
- Multiple models (herd) are better than one  
- Open models can compete with top AI  


**One-line takeaway**

“Llama 3 is a group of models trained on massive high-quality data, showing that scaling and system design matter more than new architecture.”

---
