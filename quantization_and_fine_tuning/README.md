### Quantization, KV Cache, Attention Tricks, and Fine-Tuning 

---

##  What Learnings

| Topic | What It Does | Why It Matters |
|---|---|---|
| KV Cache | Speeds up text generation | Without it, models get exponentially slower |
| Quantization | Shrinks model size | Run a 70B model on 1 GPU instead of 4+ |
| Attention Optimizations | Makes attention faster + cheaper | Longer context, higher throughput |
| LoRA / QLoRA | Fine-tunes models efficiently | Customize a model on a laptop GPU |

---

## 1. KV Cache 

### The Problem

When an LLM generates text word-by-word, it needs to look at **all previous words** every single step.

Without a cache, this gets very slow:

```
Step 1: look at 1 word      → 1 operation
Step 2: look at 2 words     → 2 operations
Step 3: look at 3 words     → 3 operations
...
Step 100: look at 100 words → 100 operations

Total = 1+2+3+...+100 = 5050 operations   (O(n²))
```

With a KV Cache:

```
Step 1: compute + SAVE word 1    → 1 operation
Step 2: compute word 2 only      → 1 operation (word 1 is cached!)
Step 3: compute word 3 only      → 1 operation
...
Step 100: compute word 100 only  → 1 operation

Total = 100 operations    (O(n))
```

### How It Works

```
┌─────────────────────────────────────────────┐
│           PREFILL PHASE                      │
│  Prompt: "The cat sat on the"               │
│                                              │
│  Process ALL words at once →                │
│  Save Keys & Values to CACHE                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           DECODE PHASE (loop)                │
│                                              │
│  New word arrives → compute its K & V       │
│  Append to cache                            │
│  Look at ALL cached K,V to decide next word │
│  Repeat...                                  │
└─────────────────────────────────────────────┘
```

>  **Simple analogy:** It's like taking notes while reading. Instead of re-reading the whole book every time you want to remember something, you refer to your notes.

### Memory Warning

KV Cache can get **huge** for long conversations:

```
Llama 70B, 128K context length:
  KV Cache = ~343 GB  ← more than the model itself!

With GQA optimization:
  KV Cache = ~43 GB   ← 8× smaller 
```

---

## 2. Quantization — Fit Big Models in Small Spaces

### The Idea

Neural network weights are just numbers. By default they use 32 bits each. We can store them using fewer bits — with a small quality trade-off.

```
FP32  ████████████████████████████████  32 bits  →  4 bytes/param
FP16  ████████████████                  16 bits  →  2 bytes/param
INT8  ████████                           8 bits  →  1 byte/param
INT4  ████                               4 bits  →  0.5 bytes/param
```

### Memory Savings Table

| Model | FP32 | FP16 | INT8 | INT4 |
|---|---|---|---|---|
| Llama 8B  | 32 GB | 16 GB | 8 GB | **4 GB** |
| Llama 70B | 280 GB | 140 GB | 70 GB | **35 GB** |
| Llama 405B | 1,620 GB | 810 GB | 405 GB | **~203 GB** |

>  INT4 = run a 70B model on **one** 48GB GPU. Without quantization you'd need 4+ GPUs.

### Number Formats Explained

```
FP32  [sign][8 exp bits][23 mantissa bits]  → very precise, very large
BF16  [sign][8 exp bits][7 mantissa bits]   → same range as FP32, half the size
FP16  [sign][5 exp bits][10 mantissa bits]  → smaller range, can overflow
INT8  [8-bit integer -128 to 127]           → needs a "scale" number too
INT4  [4-bit integer -8 to 7]               → only 16 possible values!
```

### Main Quantization Methods

#### GPTQ — Layer-by-Layer Compression
```
For each layer in the model:
  1. Run some sample text through it
  2. Quantize one column of weights at a time
  3. Measure the error caused
  4. Adjust remaining columns to compensate
  
Result: INT4 model with minimal quality loss
Time: ~3-4 hours for a 70B model
```

#### AWQ — Protect the Important Weights
```
Key insight: ~1% of weights matter WAY more than the rest

Steps:
  1. Find which weights are most "important" (high activation)
  2. Scale those up before quantizing (so they get more precision)
  3. Scale the output back down (math stays the same)
  
Result: Often better quality than GPTQ, and faster to compute
```

#### GGUF (for llama.cpp — run on CPU!)

| Format | Bits/Weight | Quality | Speed |
|---|---|---|---|
| Q2_K | 2.5 | Poor | Blazing fast |
| Q4_K_M | 4.6 | **Good**  | Fast |
| Q6_K | 6.6 | Excellent | Moderate |
| Q8_0 | 8.5 | Near perfect | Slower |

> Q4_K_M is the most popular — best balance of size and quality.

---

## 3. Attention Optimizations

### FlashAttention — Same Math, Way Less Memory

Standard attention has to create a **giant matrix** (sequence × sequence) in slow GPU memory (HBM):

```
Standard Attention:
  GPU Slow Memory (HBM)
  ┌──────────────────┐
  │  Q  K  V         │  ← load
  │  S = Q×Kᵀ        │  ← write HUGE n×n matrix to HBM 
  │  P = softmax(S)  │  ← read it back
  │  O = P×V         │  ← final output
  └──────────────────┘
  Memory: O(n²)
```

FlashAttention does the same math in **tiles** using fast on-chip memory:

```
FlashAttention:
  GPU Fast Memory (SRAM)
  ┌──────────────┐
  │ small Q tile │
  │ small K tile │  ← compute attention piece-by-piece
  │ small V tile │     the big n×n matrix NEVER exists!
  └──────────────┘
  Memory: O(n)   → 2–4× faster, identical results 
```

### GQA & MQA — Fewer KV Heads

In standard attention, every "head" has its own K and V. We can share them:

```
Multi-Head Attention (MHA) — no sharing:
  Q1→K1,V1   Q2→K2,V2   Q3→K3,V3   Q4→K4,V4
  4 KV heads = 100% cache size

Grouped Query Attention (GQA) — groups share:
  Q1,Q2→K1,V1     Q3,Q4→K2,V2
  2 KV heads = 50% cache size (used by Llama 2/3, Mistral)

Multi-Query Attention (MQA) — everyone shares:
  Q1,Q2,Q3,Q4 → K1,V1
  1 KV head = 25% cache size (used by Falcon)
```

### Sliding Window Attention (Mistral)

Instead of attending to ALL previous tokens, only look at the last W tokens:

```
Sequence: t1  t2  t3  t4  t5  t6  t7  t8
Window W = 4

t8 attends to: [t5, t6, t7, t8]  only!
               t1-t4 are forgotten from cache

Cache size = always W = constant memory 
```

> Information can still flow across long distances through many layers. With 32 layers and W=4096, effective reach = 32 × 4096 = 131K tokens.

### PagedAttention (vLLM)

Like virtual memory in an OS, but for KV cache:

```
OLD WAY (wasteful):
  Request A: needs 500 tokens → allocates 2048 → 75% wasted 
  Request B: needs 100 tokens → allocates 2048 → 95% wasted 

PagedAttention:
  Split memory into small "pages"
  Each request only uses exactly the pages it needs
  Pages are reused when a request finishes

  Req A: [page1][page2]
  Req B: [page3]
  Free:  [page4][page5][page6]...
  
  ~0% waste  → fit way more requests at once
```

---

## 4. Fine-Tuning — Teaching the Model New Tricks

### The Options

```
Full Fine-Tuning:  train ALL weights    → best quality, needs huge GPU
LoRA:              train tiny adapters  → almost as good, small GPU
QLoRA:             LoRA on 4-bit model  → good quality, fits on laptop GPU
```

### LoRA — Low-Rank Adaptation

**The insight:** When you fine-tune a model, the *changes* to the weights are surprisingly simple (low-rank). So instead of changing all weights, we add tiny "adapter" layers.

```
Normal layer:  [4096 × 4096 weight matrix]
               = 16,777,216 numbers to train 

LoRA:          [4096 × 4096 frozen weights]  ← NOT trained
             + [4096 × 16] × [16 × 4096]    ← trained instead
               = 131,072 numbers             (128× fewer!)

Forward pass:
  output = (frozen W × input) + (B × A × input)
                                ↑ this is the LoRA part
```

#### LoRA Settings

| Setting | Meaning | Recommended |
|---|---|---|
| `r` (rank) | Size of adapters | 8–64 (16 is safe default) |
| `alpha` | Scaling strength | Usually 2× rank |
| `target_modules` | Which layers to adapt | All linear layers for best results |
| `dropout` | Regularization | 0.05–0.1 |

### QLoRA — LoRA on a Diet

Combine 4-bit quantization with LoRA:

```
┌──────────────────────────────────────────┐
│  Base Model (FROZEN, 4-bit INT4)         │
│  ~0.5 bytes per parameter                │
│                ↓ dequantize              │
│         compute in BF16                  │
│                +                         │
│  LoRA Adapters (TRAINABLE, BF16)         │
│  A matrix  ×  B matrix                  │
│                =                         │
│         final output                     │
└──────────────────────────────────────────┘

Memory for Llama 70B:
  Full fine-tune:  ~1,260 GB  (impossible for most!)
  LoRA only:       ~140 GB    (still expensive)
  QLoRA:           ~35 GB     → fits on ONE 48GB GPU 
```

---

## 5. When to Use What?

```
┌─────────────────────────────────────────────────────────┐
│  Do you need custom behavior?                           │
│                                                          │
│  NO → just use the model as-is with good prompts        │
│                                                          │
│  YES ↓                                                   │
│                                                          │
│  Do you need fresh/updatable knowledge?                  │
│                                                          │
│  YES → RAG (Retrieval Augmented Generation)             │
│         Good for: docs, search, Q&A, citations          │
│                                                          │
│  NO → Fine-Tuning                                        │
│         Good for: custom style, domain expertise         │
│                                                          │
│  BOTH → Fine-tune the model AND add RAG              │
└─────────────────────────────────────────────────────────┘
```

### Quick Comparison

| | Prompt Engineering | RAG | Fine-Tuning |
|---|---|---|---|
| Setup effort |  Easy | Medium |  Hard |
| Custom style |  Limited |  Limited | Strong |
| Cost per request |  Higher |  Medium |  Lower |
| Hallucination risk |  High |  Low |  Medium |
| Data needed | None | Documents | Labeled examples |

---

##  The Big Picture

```
You have a large model (e.g., Llama 70B, 140 GB in FP16)
                    ↓
         Apply INT4 Quantization
                    ↓
         Now it's ~35 GB — fits on 1 GPU
                    ↓
         Run inference with:
           • KV Cache     → fast generation
           • FlashAttention → memory-efficient attention  
           • GQA           → smaller KV cache
           • PagedAttention (vLLM) → serve many users at once
                    ↓
         Need custom behavior?
           • QLoRA fine-tune on your data (< 8 GB VRAM!)
           • Merge adapters back into model
           • Re-quantize to GGUF for deployment
                    ↓
         Ship it 
```

---

##  Key Libraries

| Library | What It Does |
|---|---|
| `bitsandbytes` | INT4/INT8 quantization in Python |
| `peft` | LoRA and other PEFT methods |
| `trl` | Training with SFTTrainer |
| `vllm` | High-throughput serving with PagedAttention |
| `llama.cpp` | Run GGUF models on CPU or GPU |
| `transformers` | Load and run HuggingFace models |
| `flash-attn` | FlashAttention implementation |

---

