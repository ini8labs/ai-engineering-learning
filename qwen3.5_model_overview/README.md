# Qwen 3.5 Model Overview — LLM Benchmarking & Technical Learnings

**Author:** Soumili Jana Ghosh


## Overview

Qwen is a family of large language models developed by the Chinese technology company Alibaba Cloud. These models are designed to understand and generate human-like text, similar to models like OpenAI’s GPT-4 and Meta’s LLaMA.

Qwen models are built using the Transformer decoder architecture, which allows the model to process and understand language efficiently. They are trained on large amounts of text data so they can perform many tasks such as answering questions, writing content, coding, translation, and reasoning.

The Qwen model family includes different versions with various sizes and capabilities. Examples include Qwen 1, Qwen 2, Qwen 2.5, and Qwen 3.5, with model sizes ranging from small models (for faster and cheaper use) to very large models (for high performance).

These models support advanced features such as:

(1) Long context understanding

(2) Multilingual capabilities

(3) Code generation

(4) Reasoning and problem solving

(5) Multimodal abilities (text, images, etc.)

Because of these capabilities, Qwen models are widely used in AI assistants, research, enterprise applications, and software development tools.
Many developers and researchers use Qwen models because they are open-source friendly and easy to integrate with frameworks like **Hugging Face Transformers**.
The official website for the Qwen model developed by Alibaba Cloud is:
Website:
**https://qwen-ai.com**
This website provides information about Qwen models, documentation, model releases, benchmarks, and how developers can use the models.

**Qwen models come in different sizes:**

0.5B parameters

1.8B parameters

2.7B parameters

3.14B parameters

4.32B+ parameters

This allows users to choose models based on hardware and performance needs.

Qwen models perform strongly on benchmark tests such as:

(1) **MMLU**

(2) **Coding benchmarks**

(3) **Reasoning tests**

They compete with models like **GPT-4** and **LLaMA**.

**Advantages:**

(1) Strong performance

(2) Multiple model sizes

(3) Multilingual support

(4) Open model ecosystem

**Limitations:**

(1) Large models need high compute

(2) Possible hallucinations

(3) Fine-tuning can be expensive

**Real World Applications:**

Qwen models are used in:

(1) AI chatbots

(2) Coding assistants

(3) Document analysis

(4) Enterprise AI systems

(5) Research and development

This project presents a hands-on study of the **Qwen 2.5-0.5B** large language model, focusing on running it locally on CPU, benchmarking its performance, and exploring various optimization techniques.


## Objectives

- Understand the architecture of Qwen 3.5
- Run the model locally on CPU
- Perform LLM benchmarking (inference time, RAM usage, token generation speed)
- Explore techniques to optimize model performance


##  Tools & Environment

| Category            | Tool / Library                      |
|---------------------|-------------------------------------|
| Programming Language | **Python**                             |
| Model Loading       | **Hugging Face Transformers**           |
| Deep Learning Backend | **PyTorch**                           |
| Performance Monitoring | **psutil**                           |
| Compute Environment | **Google Colab**                        |

**Installation:**
```bash
pip install transformers accelerate psutil
```

##  How an LLM Works (Summary)

1. **Tokenization & Embedding** — Text is converted to tokens, then to numerical vectors.
2. **Transformer Layers** — Decoder-only architecture (like GPT/LLaMA) processes relationships between tokens via self-attention and feed-forward networks.
3. **Next Token Prediction** — The model repeatedly predicts the next token to generate text.



## Benchmarking Pipeline

```
Load Model → Prepare Prompt → Tokenize Input → Run Inference → Measure Performance
```

### Metrics Measured

| Metric | Method |
|--------|--------|
| Inference time | `time.time()` |
| RAM usage | `psutil.Process().memory_info().rss` |
| Token generation speed | `tokens_generated / total_time` |


##  Optimization Experiments

**|#| Experiment -- Outcome**



| 1 | Reduce `max_new_tokens` (100 → 50) -- Faster inference 

| 2 | Deterministic decoding (`do_sample=False`) -- Consistent, predictable output 

| 3 | Use smaller Qwen model variants -- Faster on CPU 

| 4 | `use_cache=True` -- Avoids recomputing attention 

| 5 | `torch.compile(model)` -- Merges operations, reduces overhead 

| 6 | Batch small prompts -- Better hardware utilization 

> **Other experiments explored:** 

float16 on CPU -- Often slower. CPU doesn’t accelerate FP16 well.

pruning heads --  Requires retraining or structured pruning tools.

vocabulary limiting -- Requires logits processor customization.

cache reuse for repeated prompts --  used in production servers.

ONNX (limited by RAM)



##  Technical Concepts Learned

- **Transformer Architecture** — Decoder-only models (similar to GPT)
- **RoPE** (Rotary Position Embedding) — Position-aware vector rotation
- **RMSNorm** (Root Mean Square Normalization) — Faster alternative to LayerNorm
- **MoE** (Mixture of Experts) — Parallel expert networks for faster, specialized inference
- **Multimodal** — Handling text, audio, video
- **Gated Delta Net** — Improved long-sequence learning
- **GQA** (Grouped Query Attention) — Shared KV heads across query vectors
- **Linear Attention** — O(n) attention instead of O(n²)
- **MTP** (Multi Token Prediction) — Predicts multiple future tokens in one step



##  Challenges

- High RAM usage when running on CPU
- Slow inference speed in some configurations

**Mitigations used:** smaller models, reduced token generation, deterministic mode, `use_cache=True`, prompt batching.


##  Future Work

- Run experiments on **GPU**
- Explore **ONNX runtime** optimization

---

##  Conclusion

- Successfully ran Qwen 3.5 locally on CPU
- Gained practical understanding of LLM inference
- Measured real performance metrics through experiments
- Identified actionable methods to improve speed and efficiency