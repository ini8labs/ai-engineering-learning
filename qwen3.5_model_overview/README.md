# Qwen 3.5 Model Overview — LLM Benchmarking & Technical Learnings

**Author:** Soumili Jana Ghosh

---

##  Table of Contents

1. [What is Qwen?](#what-is-qwen)
2. [What is Alibaba Cloud?](#what-is-alibaba-cloud)
3. [What is a Large Language Model (LLM)?](#what-is-a-large-language-model-llm)
4. [Qwen Model Family](#qwen-model-family)
5. [What are Model Parameters?](#what-are-model-parameters)
6. [Key Features of Qwen](#key-features-of-qwen)
7. [Benchmarks — How We Measure AI Performance](#benchmarks--how-we-measure-ai-performance)
   - [MMLU](#1-mmlu--massive-multitask-language-understanding)
   - [Coding Benchmarks](#2-coding-benchmarks-humaneval--mbpp)
   - [Reasoning Tests](#3-reasoning-tests-gsm8k--arc)
   - [Benchmark Summary Table](#benchmark-summary-table)
8. [Advantages of Qwen](#advantages-of-qwen)
9. [Limitations of Qwen](#limitations-of-qwen)
10. [Real World Applications](#real-world-applications)
11. [Project Objectives](#project-objectives)
12. [Tools & Environment](#tools--environment)
13. [Installation & Requirements](#installation--requirements)
14. [Project Code Structure](#project-code-structure)
15. [Benchmarking Pipeline](#benchmarking-pipeline)
16. [Optimization Experiments](#optimization-experiments)
17. [Technical Concepts Learned](#technical-concepts-learned)
18. [Challenges & Mitigations](#challenges--mitigations)
19. [Future Work](#future-work)
20. [Conclusion](#conclusion)

---

## What is Qwen?

**Qwen** (pronounced "Chwen") is a family of **large language models (LLMs)** developed by **Alibaba Cloud**, the cloud computing division of Alibaba Group — one of the world's largest technology companies, based in China.

These models are designed to **understand and generate human-like text**, making them capable of tasks such as:
- Answering questions
- Writing articles or code
- Translating languages
- Summarizing documents
- Logical reasoning and problem solving

Qwen is comparable to well-known models like **OpenAI's GPT-4** and **Meta's LLaMA**, but with a strong focus on being **open-source friendly** and supporting **multiple languages**, especially Chinese and English.

>  Official Website: [https://qwen-ai.com](https://qwen-ai.com)

---

## What is Alibaba Cloud?

Alibaba Cloud (also known as **Aliyun**) is the cloud computing arm of Alibaba Group, founded in 2009. It is one of the **top 3 largest cloud providers in the world**, alongside AWS (Amazon) and Azure (Microsoft).

Alibaba Cloud provides infrastructure, AI services, databases, and machine learning tools to businesses globally. Developing Qwen is part of their mission to advance AI research and make powerful AI accessible.

---

## What is a Large Language Model (LLM)?

A **Large Language Model (LLM)** is a type of AI model trained on massive amounts of text data. It learns the patterns, grammar, facts, and reasoning found in that text, so it can generate new text that is coherent and contextually appropriate.

| Concept | Simple Explanation |
|---|---|
| **Training data** | Billions of pages of text from the internet, books, code, etc. |
| **Parameters** | Numbers inside the model that store learned knowledge |
| **Inference** | When the model uses its knowledge to generate a response |
| **Token** | A small chunk of text (roughly a word or part of a word) |

> For further details about how LLMs work internally, refer to: `ai-engineering-learning/qwen3.5_model_overview/llm.md`

---

## Qwen Model Family

The Qwen family has evolved through multiple generations, each improving on the last:

| Version | Key Highlights |
|---|---|
| **Qwen 1** | First generation, basic language understanding |
| **Qwen 2** | Improved multilingual support, better reasoning |
| **Qwen 2.5** | Enhanced coding, math, and instruction following |
| **Qwen 3.5** | Latest generation with advanced reasoning and multimodal features |

Each version also comes in **different sizes** — from tiny models you can run on a laptop, to massive models that need powerful servers.

---

## What are Model Parameters?

**Parameters** are the numbers (weights) inside a neural network that determine how it responds to input. More parameters generally means:
- More knowledge stored
- Better performance on complex tasks
- But also more memory and computing power needed

Think of parameters like the neurons in a brain — more neurons can mean more intelligence, but also more energy needed.

### Qwen Model Sizes

| Model Size | Best For | Hardware Needed |
|---|---|---|
| **0.5B** (500 million) | Quick experiments, low-resource devices | Laptop CPU |
| **1.8B** | Lightweight applications | Laptop / Budget GPU |
| **2.7B** | Balanced performance | Mid-range GPU |
| **3.14B** | Good general performance | GPU with 8GB+ VRAM |
| **4.32B+** | High accuracy tasks | GPU with 16GB+ VRAM |
| **7B** | Near GPT-3.5 quality | GPU with 16–24GB VRAM |
| **72B** | Near GPT-4 quality | Multiple GPUs / Cloud |

> This project uses **Qwen 2.5-0.5B** — the smallest variant — to run locally on CPU without needing any GPU.

---

## Key Features of Qwen

### 1. Long Context Understanding
Qwen can read and understand very long documents — up to tens of thousands of words — without losing track of earlier information. This is useful for legal documents, books, or long codebases.

### 2. Multilingual Capabilities
Qwen supports many languages including English, Chinese, French, Spanish, Arabic, and more. This makes it valuable for global applications.

### 3. Code Generation
Qwen can write, explain, debug, and complete code in languages like Python, JavaScript, Java, C++, and more. Specialized versions (Qwen-Coder) are fine-tuned specifically for coding.

### 4. Reasoning and Problem Solving
Qwen can follow multi-step logical chains to solve math problems, puzzles, and complex instructions — not just recall facts but actually "think through" problems.

### 5. Multimodal Abilities
Newer Qwen models (Qwen-VL, Qwen-Audio) can process **images, audio, and video** in addition to text, enabling richer real-world applications.

---

## Benchmarks — How We Measure AI Performance

**Benchmarks** are standardized tests that measure how well an AI model performs across different skills. Think of them like school exams — each one tests a different "subject."

A score alone means little without context, so benchmarks compare models against each other and against human-level performance.

---

### 1. MMLU — Massive Multitask Language Understanding

**What it is:**
MMLU is a comprehensive multiple-choice test covering **57 different academic subjects**, ranging from elementary math to professional law and medicine. It was created by researchers at UC Berkeley to measure the breadth of a model's world knowledge.

**Subjects covered include:**
- History, Geography, Economics
- Biology, Chemistry, Physics
- Law, Medicine, Psychology
- Mathematics, Computer Science, Ethics
- ... and 47 more subjects

**Why it matters:**
It answers the question: *"How knowledgeable and well-rounded is this AI model"*

A model that scores well on MMLU has genuinely learned a wide range of human knowledge — not just one narrow domain.

**Scoring:** Percentage of questions answered correctly (0–100%). Human expert baseline is around 89%.

| Model | MMLU Score | What It Means |
|---|---|---|
| Human Expert | ~89% | Gold standard |
| GPT-4 | ~86% | Near-expert level |
| Qwen 2.5-72B | ~85% | Competitive with GPT-4 |
| Qwen 2.5-7B | ~74% | Strong general knowledge |
| LLaMA 3-8B | ~68% | Good but weaker breadth |
| Qwen 2.5-0.5B | ~47% | Basic, limited knowledge |
| Random Guessing | ~25% | Baseline (4 answer choices) |

>  **Example MMLU question:** *"Which of the following is the primary function of the mitochondria?"*
> (A) Protein synthesis (B) Energy production (C) DNA replication (D) Lipid storage
> **Correct answer: (B)**

---

### 2. Coding Benchmarks (HumanEval / MBPP)

**What it is:**
These benchmarks give the model real programming problems and check if the code it generates actually **runs correctly and passes test cases**.

- **HumanEval** — Created by OpenAI; contains 164 handwritten Python programming challenges
- **MBPP** (Mostly Basic Python Problems) — Created by Google; contains ~374 Python tasks ranging from beginner to intermediate

**Why it matters:**
It answers the question: *"Can this AI write code that actually works?"*

Unlike MMLU (which checks knowledge), coding benchmarks check functional correctness — the code is actually executed and tested automatically.

**Scoring:** **Pass@1** = percentage of problems where the model's first attempt produces correct, working code.

| Model | HumanEval (Pass@1) | MBPP (Pass@1) |
|---|---|---|
| GPT-4 | ~87% | ~83% |
| Qwen 2.5-72B | ~86% | ~85% |
| Qwen 2.5-7B | ~79% | ~73% |
| LLaMA 3-8B | ~62% | ~67% |
| Qwen 2.5-0.5B | ~30% | ~35% |

> **Example HumanEval problem:**
> *"Write a Python function that takes a list of integers and returns only the even numbers."*
> The model writes the function → it's automatically run → test cases check if it returns correct output.

---

### 3. Reasoning Tests (GSM8K / ARC)

**What it is:**
These benchmarks test whether the model can **think step-by-step** through problems rather than just recall memorized answers.

- **GSM8K** (Grade School Math 8K) — Created by OpenAI; 8,500 grade-school math word problems that require multiple reasoning steps
- **ARC** (AI2 Reasoning Challenge) — Created by Allen Institute for AI; science questions that require genuine reasoning, not just pattern matching

**Why it matters:**
It answers the question: *"Can this AI actually reason, or does it just look things up?"*

A model that genuinely reasons can solve **new problems it has never seen before**, rather than just repeating memorized answers.

**Scoring:** Percentage of problems solved correctly.

| Model | GSM8K (Math Reasoning) | ARC-Challenge (Logic) |
|---|---|---|
| GPT-4 | ~92% | ~96% |
| Qwen 2.5-72B | ~91% | ~95% |
| Qwen 2.5-7B | ~85% | ~90% |
| LLaMA 3-8B | ~75% | ~79% |
| Qwen 2.5-0.5B | ~36% | ~60% |

>  **Example GSM8K problem:**
> *"A store sells apples for $0.50 each and oranges for $0.75 each. If Sarah buys 4 apples and 3 oranges, how much does she spend in total"*
>
> The model must reason:
> - Step 1: 4 × $0.50 = $2.00
> - Step 2: 3 × $0.75 = $2.25
> - Step 3: $2.00 + $2.25 = **$4.25**

---

### Benchmark Summary Table

| Benchmark | What It Tests | Analogy | Scale |
|---|---|---|---|
| **MMLU** | World knowledge across 57 subjects | University entrance exam | 0–100% |
| **HumanEval** | Writing correct Python code | Coding interview | Pass@1 % |
| **MBPP** | Basic to intermediate Python tasks | Coding homework | Pass@1 % |
| **GSM8K** | Multi-step math word problems | Grade school math test | 0–100% |
| **ARC** | Scientific reasoning & logic | Science Olympiad | 0–100% |

> **Key Takeaway for this project:** The Qwen 2.5-0.5B model (used here) scores modestly on all benchmarks — expected for its tiny size. It is designed for experimentation and learning on CPU, not production use. Larger Qwen models (7B, 72B) are genuinely competitive with GPT-4.

---

## Advantages of Qwen

| Advantage | Details |
|---|---|
| **Strong performance** | Competitive with GPT-4 at larger sizes |
| **Multiple model sizes** | From 0.5B (laptop) to 72B+ (server) |
| **Multilingual support** | Strong Chinese and English, plus many others |
| **Open model ecosystem** | Available on Hugging Face, easy to download and use |
| **Free to use** | Most variants are free for research and development |
| **Active development** | Alibaba regularly releases improved versions |

---

## Limitations of Qwen

| Limitation | Details |
|---|---|
| **Large models need high compute** | 72B models require expensive multi-GPU setups |
| **Possible hallucinations** | Like all LLMs, can generate confident but wrong answers |
| **Fine-tuning can be expensive** | Customizing large models requires significant GPU resources |
| **Slow on CPU** | Without a GPU, inference is very slow (as seen in this project) |
| **Context window limits** | Very long documents may exceed the model's processing capacity |

---

## Real World Applications

| Application | How Qwen is Used |
|---|---|
| **AI Chatbots** | Customer service bots, virtual assistants |
| **Coding Assistants** | Auto-complete code, explain bugs, write documentation |
| **Document Analysis** | Summarize contracts, extract key information from reports |
| **Enterprise AI Systems** | Internal knowledge bases, automated workflows |
| **Research & Development** | Academic research, data analysis, hypothesis generation |
| **Translation Services** | Multilingual content, cross-language communication |
| **Education** | Tutoring systems, quiz generation, explanation tools |

---

## Project Objectives

This project provides a hands-on study of the **Qwen 2.5-0.5B** model with the following goals:

-  Understand the architecture of Qwen 3.5 at a conceptual level
-  Run the model locally on a **CPU** (no GPU required)
-  Perform **LLM benchmarking** — measuring inference time, RAM usage, and token generation speed
-  Explore and compare multiple **optimization techniques** to improve performance

---

## Tools & Environment

| Category | Tool / Library | Purpose |
|---|---|---|
| Programming Language | **Python** | Main language for all scripts |
| Model Loading | **Hugging Face Transformers** | Download and run Qwen models easily |
| Deep Learning Backend | **PyTorch** | Underlying framework that runs the model |
| Performance Monitoring | **psutil** | Measure RAM usage during inference |
| Compute Environment | **Google Colab** | Free cloud notebook with CPU/GPU access |

---

## Installation & Requirements

### Quick Install
```bash
pip install transformers accelerate psutil
```

### Full Requirements
All dependencies are listed in:
```
ai-engineering-learning/qwen3.5_model_overview/qwen3,5_model_overview/requirement.txt
```

Install everything at once with:
```bash
pip install -r requirements.txt
```

---

## Project Code Structure

```
ai-engineering-learning/
└── qwen3.5_model_overview/
    ├── Qwen2.5_0_5B.ipynb     ← Main notebook (model loading, benchmarking, experiments)
    ├── llm.md                  ← Deep dive into how LLMs work
    └── requirement.txt         ← All Python dependencies
```

### What's inside `Qwen2.5_0_5B.ipynb`?

| Section | What It Does |
|---|---|
| Model Loading | Downloads and loads Qwen 2.5-0.5B using `AutoTokenizer` and `AutoModelForCausalLM` |
| Benchmarking Pipeline | Measures inference time, RAM usage, and tokens/second |
| Optimization Experiments | Tests 6+ techniques to make the model faster |

---

## Benchmarking Pipeline

The pipeline follows these steps in order:

```
Load Model → Prepare Prompt → Tokenize Input → Run Inference → Measure Performance
```

### Step-by-Step Explanation

| Step | What Happens | Code Used |
|---|---|---|
| **Load Model** | Download weights from Hugging Face and load into memory | `AutoModelForCausalLM.from_pretrained(...)` |
| **Prepare Prompt** | Write the input text you want the model to respond to | Python string |
| **Tokenize Input** | Convert text into numbers the model understands | `AutoTokenizer(...)` |
| **Run Inference** | The model generates its response token by token | `model.generate(...)` |
| **Measure Performance** | Record how long it took, how much RAM was used, how fast tokens were generated | `time.time()`, `psutil` |

### Metrics Measured

| Metric | What It Tells You | How It's Measured |
|---|---|---|
| **Inference time** | How many seconds to generate a response | `time.time()` before and after |
| **RAM usage** | How much memory the model consumes | `psutil.Process().memory_info().rss` |
| **Token generation speed** | How many tokens (words) generated per second | `tokens_generated / total_time` |

---

## Optimization Experiments

These experiments test different techniques to make the model run faster or more efficiently on CPU.

### Core Experiments

| # | Experiment | What Was Changed | Outcome |
|---|---|---|---|
| 1 | **Reduce `max_new_tokens`** | Lowered from 100 → 50 tokens |  Faster inference — model stops sooner |
| 2 | **Deterministic decoding** | Set `do_sample=False` | Consistent, predictable output every run |
| 3 | **Smaller model variants** | Used 0.5B instead of 1.8B+ |  Much faster on CPU with lower RAM |
| 4 | **`use_cache=True`** | Enabled KV caching |  Avoids recomputing attention for each token |
| 5 | **`torch.compile(model)`** | Compiled model graph |  Merges operations, reduces overhead |
| 6 | **Batch small prompts** | Grouped multiple prompts together |  Better hardware utilization |

### Additional Experiments Explored

| Experiment | Result | Why |
|---|---|---|
| **float16 on CPU** | Often slower  | CPUs are not optimized to accelerate FP16 math |
| **Pruning attention heads** | Requires retraining  | Needs structured pruning tools and fine-tuning |
| **Vocabulary limiting** | Complex to implement  | Requires custom logits processor |
| **Cache reuse for repeated prompts** | Effective  | Used in production servers for efficiency |
| **ONNX export** | Limited by RAM  | Conversion works but memory was a constraint |

---

## Technical Concepts Learned

These are the core architectural ideas behind how Qwen (and most modern LLMs) work under the hood:

| Concept | What It Is | Why It Matters |
|---|---|---|
| **Transformer Architecture** | Decoder-only design (like GPT/LLaMA) that processes tokens with attention | Foundation of all modern LLMs |
| **RoPE** (Rotary Position Embedding) | Encodes token position using vector rotation instead of fixed embeddings | Helps the model understand word order and handle long contexts |
| **RMSNorm** (Root Mean Square Normalization) | Normalizes layer values using RMS instead of mean/variance | Faster and more stable than traditional LayerNorm |
| **MoE** (Mixture of Experts) | Routes each token through only a subset of specialist sub-networks | More efficient — not all parameters activate for every input |
| **Multimodal** | Ability to process text, images, audio, and video together | Enables richer, real-world AI applications |
| **Gated Delta Net** | Architecture improvement for long-sequence learning | Better memory of distant context in long documents |
| **GQA** (Grouped Query Attention) | Multiple query heads share the same key-value heads | Reduces memory usage while maintaining performance |
| **Linear Attention** | Approximates attention in O(n) instead of O(n²) | Dramatically faster for very long sequences |
| **MTP** (Multi Token Prediction) | Predicts multiple future tokens in a single step | Speeds up generation without losing quality |

---

## Challenges & Mitigations

| Challenge | Impact | Mitigation Used |
|---|---|---|
| **High RAM usage on CPU** | Model may crash or slow down the system | Used smaller 0.5B model; monitored with psutil |
| **Slow inference speed** | Long wait times for responses | Reduced `max_new_tokens`, enabled `use_cache=True`, used deterministic decoding |
| **No GPU available** | Can't use hardware acceleration | Ran on CPU only; chose smallest model size |
| **Batching complexity** | Harder to implement correctly | Tested basic batching with small prompt groups |

---

## Future Work

| Planned Experiment | Goal |
|---|---|
| **Run on GPU** | Compare GPU vs CPU inference speed — expected 10–50x speedup |
| **ONNX Runtime optimization** | Export model to ONNX format for faster CPU inference |
| **Larger model testing** | Run Qwen 2.5-7B on GPU to see quality improvements |
| **Fine-tuning experiments** | Adapt Qwen to a specific domain (e.g., medical or legal text) |
| **Quantization (INT4/INT8)** | Compress model to run faster with less RAM |

---

## Conclusion

This project successfully demonstrated:

| Achievement | Details |
|---|---|
|  **Ran Qwen locally on CPU** | Loaded and generated text with Qwen 2.5-0.5B without any GPU |
|  **Practical benchmarking** | Measured real inference time, RAM usage, and token speed |
|  **Optimization experiments** | Tested 6+ techniques and identified what actually improves performance |
|  **Deep technical learning** | Understood RoPE, GQA, MoE, MTP, and other modern LLM components |
|  **Clear performance insights** | Identified that `use_cache=True`, smaller models, and fewer tokens are the most impactful CPU optimizations |

---

*For LLM architecture deep-dive, see:* `ai-engineering-learning/qwen3.5_model_overview/llm.md`

*For requirements, see:* `ai-engineering-learning/qwen3.5_model_overview/requirement.txt`

*For model experiment, see:* `ai-engineering-learning/qwen3.5_model_overview/Qwen2_5_0_5B.ipynb`