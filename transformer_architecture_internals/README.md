# Week 3: Transformer Architecture



---

##  Table of Contents

| # | Topic | What You'll Learn |
|---|-------|------------------|
| 1 | [The Transformer Architecture](#1-the-transformer-architecture) | The big picture — what it is and why it replaced older models |
| 2 | [Multi-Head Attention](#2-multi-head-attention) | How the model looks at a sentence from multiple angles simultaneously |
| 3 | [Q, K, V Matrices](#3-q-k-v-matrices-deep-dive) | The search-engine mechanism that powers attention |
| 4 | [Cross-Attention](#4-cross-attention) | How a translation model reads the source language while writing the target |
| 5 | [Feed-Forward Networks](#5-feed-forward-neural-networks) | The "thinking layer" that processes what attention collected |
| 6 | [Layer Normalization](#6-layer-normalization) | How the model keeps its numbers stable during training |
| 7 | [Residual Connections](#7-residual-connections) | The trick that makes 100+ layer networks trainable |
| 8 | [Logits and Output](#8-logits-and-output) | How the model decides which word to write next |
| 9 | [Summary and Exercises](#week-3-summary) | Key points and practice problems |

---

## 1. The Transformer Architecture

###  Why Does This Matter?

In 2017, a team of Google researchers published a paper titled **"Attention Is All You Need"**. At
the time, it looked like just another research paper. In hindsight, it was one of the most
consequential papers in computer science history. Every major AI system you interact with today —
ChatGPT, Claude, Gemini, Llama, GitHub Copilot, Google Translate, and hundreds more — is built on
the architecture described in that 8-page paper.

Understanding the Transformer is understanding the foundation of modern AI.

---

### The Two Parts: Encoder and Decoder

The original Transformer has two main components. Think of them like two specialists working
together on a translation job:

```
+-----------------------------------------------------------------------------+
|                        THE TRANSFORMER ARCHITECTURE                         |
|                                                                              |
|   +======================+               +======================+            |
|   |      ENCODER         |               |       DECODER        |            |
|   |  (The Reader)        |               |  (The Writer)        |            |
|   +======================+               +======================+            |
|   |                      |               |                      |            |
|   |  +----------------+  |               |  +----------------+  |            |
|   |  | Feed-Forward   |  |               |  |  Feed-Forward  |  |            |
|   |  | Network        |  |               |  |  Network       |  |            |
|   |  +-------+--------+  |               |  +-------+--------+  |            |
|   |      Add & Norm      |               |      Add & Norm      |            |
|   |  +----------------+  |  --- K,V ---> |  +----------------+  |            |
|   |  | Self-Attention |  |               |  | Cross-Attention|  |            |
|   |  +-------+--------+  |               |  +-------+--------+  |            |
|   |      Add & Norm      |               |      Add & Norm      |            |
|   |                      |               |  +----------------+  |            |
|   |  [Repeated N times]  |               |  | Masked         |  |            |
|   +======================+               |  | Self-Attention |  |            |
|   |  Input Embeddings    |               |  +-------+--------+  |            |
|   |  + Positional Enc.   |               |      Add & Norm      |            |
|   +======================+               |  [Repeated N times]  |            |
|                                          +======================+            |
|   Input: "The cat sat"                   |  Output Embeddings   |            |
|   (your source sentence)                 |  + Positional Enc.   |            |
|                                          +======================+            |
|                                                                              |
|                                  Output: "Le chat etait assis"              |
+-----------------------------------------------------------------------------+
```

**The Encoder — The Reader:**
The encoder reads the entire input sequence all at once and builds a deep, rich understanding of
it. It does not produce words — it produces a packed, information-dense representation of the
input. Think of it as the part of your brain that reads a sentence in English and fully grasps
what it means before you start translating.

**The Decoder — The Writer:**
The decoder takes the encoder's understanding and generates the output one token at a time. Each
time it produces a word, it looks back at everything it has produced so far AND at the encoder's
rich representation of the source. It is like the part of your brain that, having understood the
English sentence, now writes it out in French.

---

### Three Kinds of Transformers

Over the years, researchers discovered you do not always need both halves. Depending on the task,
you might only need one:

```
+===============+======================================+=================================+
|     TYPE      |            STRUCTURE                 |       BEST USE CASES            |
+===============+======================================+=================================+
|               | Only the Encoder (left side).        | - Sentence classification        |
| Encoder-Only  | Every token can attend to EVERY      | - Named entity recognition       |
| (BERT, etc.)  | other token in both directions.      | - Sentence similarity            |
|               | Called "bidirectional attention".    | - Question answering             |
+---------------+--------------------------------------+---------------------------------+
|               | Only the Decoder (right side).       | - Text generation (chatbots)     |
| Decoder-Only  | Each token can only attend to        | - Code generation (Copilot)      |
| (GPT, Claude, | tokens BEFORE it (left-to-right).    | - Any task framed as text output |
| Llama, etc.)  | Called "causal attention".           | - Creative writing, reasoning    |
+---------------+--------------------------------------+---------------------------------+
| Encoder-      | Both halves. Encoder reads source.   | - Machine translation            |
| Decoder       | Decoder writes output while          | - Document summarization         |
| (T5, BART,    | attending to encoder via             | - Speech-to-text (Whisper)       |
| Whisper)      | cross-attention.                     | - Any sequence-to-sequence task  |
+===============+======================================+=================================+
```

>  **Why has Decoder-Only become the dominant design?**
>
> GPT-4, Claude, Llama, and Gemini are all decoder-only. They win on almost every task by simply
> framing everything as text generation. Want to classify sentiment? Generate "positive" or
> "negative". Want to solve math? Generate the solution steps. Want code? Generate code.
> This universal flexibility, combined with proven scaling to hundreds of billions of parameters,
> is why decoder-only took over frontier AI by 2023 and has stayed dominant.

---

### The Problem With the Old Way (RNNs)

Before Transformers, we used Recurrent Neural Networks (RNNs) and their improved version, LSTMs.
Here is the fundamental bottleneck that held them back:

```
RNN PROCESSING (sequential — like reading one word at a time, slowly):

Sentence: "The fluffy cat who lived next door sat on the mat"

  Word 1   Word 2   Word 3   Word 4  ...  Word 10  Word 11
  "The" -> "fluffy" -> "cat" -> "who" -> ... -> "the" -> "mat"
    |          |          |        |                |        |
   [h1] ---> [h2] ----> [h3] --> [h4] --> ... --> [h10]--> [h11]
                                                             |
                                                          Output

PROBLEM 1 — SLOW:
  Word 5 cannot be processed until Words 1-4 are done.
  Word 11 cannot be processed until Words 1-10 are done.
  The entire computation is sequential. GPUs are wasted — they are built
  for parallelism but are forced to sit idle waiting for previous words.

PROBLEM 2 — FORGETTING:
  By the time we process "mat" (word 11), the hidden state h11 has passed
  through 10 transformations since "cat" (word 3). Information about "cat"
  has been overwritten and diluted. Early words effectively disappear.
  LSTMs help, but do not fully solve this — they just forget more slowly.

PROBLEM 3 — VANISHING GRADIENTS:
  When training, error signals must travel backward through all 11 steps.
  Each step multiplies the gradient by a small number. After 11 steps,
  the signal for early words becomes nearly zero. They barely learn.
```

```
TRANSFORMER PROCESSING (parallel — all words at once):

Sentence: "The fluffy cat who lived next door sat on the mat"

  "The"    "fluffy"  "cat"    "who"   "lived"  "next"  "door"  "sat"  "on"  "the"  "mat"
    |          |        |        |        |        |        |       |      |      |      |
    +----------+--------+--------+--------+--------+--------+-------+------+------+------+
    |                                                                                     |
    |                       ATTENTION LAYER (processes ALL words simultaneously)          |
    |                       Every word directly compares itself to every other word.      |
    |                       Distance does not matter. "mat" and "cat" talk directly!      |
    |                                                                                     |
    +----------+--------+--------+--------+--------+--------+-------+------+------+------+
    |          |        |        |        |        |        |       |      |      |      |

BENEFIT 1 — SPEED: All words processed in parallel. Full GPU utilization.
BENEFIT 2 — NO FORGETTING: Direct attention between ANY two positions, regardless of distance.
BENEFIT 3 — SCALES: Proven to scale to hundreds of billions of parameters with no fundamental limit.
```

| Feature | RNNs / LSTMs | Transformers |
|---------|-------------|--------------|
| Processing order | Sequential (word by word) | Fully parallel (all at once) |
| Long-range memory | Poor — degrades over distance | Perfect — direct attention between any two tokens |
| GPU utilization | Low — sequential bottleneck | Very high — massively parallel |
| Training speed | Slow | Fast |
| Maximum practical scale | ~1 billion parameters | 100s of billions (proven repeatedly) |
| Memory cost | O(1) per step | O(n²) for attention (quadratic in sequence length) |

---

## 2. Multi-Head Attention

### The Core Insight: Language Is Multidimensional

When you read the sentence **"The bank can guarantee deposits will eventually cover tuition costs
because it was announced that a large pool of money is available"**, your brain simultaneously
processes several completely different types of information:

- **Grammar:** "bank" is the subject, "guarantee" is the verb
- **Word sense disambiguation:** "bank" means a financial institution here, not a riverbank
- **Coreference:** "it" refers to "the bank" (not any of the other nouns!)
- **Long-range reasoning:** "cover" relates to "costs" which relates to "tuition"
- **Sentence structure:** "because" signals a causal clause

A single attention mechanism can only specialize in one type of relationship at a time. If it
focuses on coreference, it might miss syntactic structure. **Multi-Head Attention** solves this
by running several independent attention processes in parallel, each free to specialize in a
different type of linguistic relationship.

---

###  Visual: How Multiple Heads Work Together

```
Input: "The cat sat on the mat because it was comfortable"
                        |
                        v
          +=============================+
          |     INPUT EMBEDDING X       |
          |   (seq_len x d_model)       |
          +=============================+
                        |
         Split into H independent attention heads
          |           |           |           |
          v           v           v           v
      [HEAD 1]    [HEAD 2]    [HEAD 3]    [HEAD 4]  ...more heads
       Grammar    Coreference  Semantics   Position
      structure   resolution   similarity  patterns

      "cat" is    "it" refers  "sat" is    "mat" is
      subject     to "cat"     similar     near "on"
      of "sat"    (not mat!)   to "rest"

          |           |           |           |
          +-----+-----+-----+-----+
                      |
                      v
             Concatenate all head outputs
                      |
                      v
             Final projection (W_O)
                      |
                      v
             Rich multi-dimensional
             understanding of every
             word in context
```

Each head learns independently during training. Nobody tells head 1 to focus on grammar or head 2
to focus on coreference — they discover these specializations on their own through gradient
descent, because specializing is what minimizes the training loss.

---

###  The Formula 

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) x W_O

where each head_i = Attention( Q x W_Q_i,  K x W_K_i,  V x W_V_i )
```

Breaking this down step by step:

1. Take the input X — your sequence of word embeddings
2. For each head i, project X through three different learned weight matrices to get that head's
   own Q, K, and V matrices
3. Run scaled dot-product attention independently for each head (they do not share information
   during this step)
4. Concatenate all head outputs back into one big vector
5. Apply one final linear projection (W_O) that blends the heads' different perspectives

---

### The Attention Weight Matrix 

After computing attention, you get a matrix where each row is a token and each column is "how
much I attended to that token during this pass". Let's look at a concrete example:

```
Sentence: "The cat sat on it"

              |<------- Keys (tokens being attended to) -------->|
              The   cat   sat    on    it
            +------------------------------------------+
  The       | 0.45  0.25  0.12  0.10  0.08 |   "The" mostly looks at itself
  cat       | 0.15  0.35  0.22  0.08  0.20 |   "cat" spreads attention around
  sat       | 0.05  0.40  0.25  0.18  0.12 |   "sat" focuses on its subject "cat"
  on        | 0.03  0.10  0.42  0.30  0.15 |   "on" looks at the verb "sat"
  it        | 0.02  0.65  0.10  0.05  0.18 |   "it" STRONGLY attends to "cat"!
            +------------------------------------------+
  ^
  Queries (tokens doing the attending)

Each ROW sums to exactly 1.0 — that's what softmax enforces.
Higher number = this token has more influence on the row token's output.

The key insight: "it" has learned that it refers to "cat" in this context.
Nobody programmed this rule — the model discovered it during training because
attending to the correct antecedent helps predict downstream words correctly.
```

---

###  Exact Shape Walkthrough (Using BERT-base Numbers)

Let's trace every tensor shape precisely:

```
Configuration:
  d_model = 768   <- total embedding dimension
  n_heads = 12    <- number of attention heads
  d_k     = 64    <- dimension per head (768 / 12 = 64)
  batch   = 2     <- processing 2 sentences at once
  seq_len = 10    <- each sentence has 10 tokens

Step 1: Input arrives
  X shape: (2, 10, 768)
           ^  ^   ^
           |  |   768 numbers describe each word
           |  10 words per sentence
           2 sentences in this batch

Step 2: Project to Q, K, V using three (768, 768) weight matrices
  Q shape: (2, 10, 768)  <- "query" version of each word
  K shape: (2, 10, 768)  <- "key" version of each word
  V shape: (2, 10, 768)  <- "value" version of each word
  (same shape as input, but in different learned "spaces")

Step 3: Split into 12 heads, each head gets 64 dimensions
  Q_heads shape: (2, 12, 10, 64)
                  ^  ^   ^   ^
                  |  |   |   64 dimensions for this head
                  |  |   10 words
                  |  12 independent heads
                  2 batch examples

Step 4: Compute attention for all 12 heads IN PARALLEL
  scores shape:  (2, 12, 10, 10)  <- each head has a 10x10 attention matrix
  weights shape: (2, 12, 10, 10)  <- after applying softmax
  context shape: (2, 12, 10, 64)  <- weighted combination of values

Step 5: Concatenate all 12 heads back together
  combined shape: (2, 10, 768)    <- 12 heads * 64 dims = 768, back to original!

Step 6: Apply final output projection W_O
  output shape:   (2, 10, 768)    <- exactly same shape as the original input!

This is the critical property: input shape == output shape.
Because of this, you can stack as many transformer blocks as you want.
Each block takes (batch, seq, d_model) and returns (batch, seq, d_model).
```

---

## 3. Q, K, V Matrices Deep Dive

###  The Database Search Analogy (The Best Way to Understand Attention)

Imagine you are searching Google for restaurant recommendations:

```
YOU TYPE:   "best Italian restaurant near me"
                    ^
                    This is your QUERY (Q)
                    It represents: "what information am I looking for?"

GOOGLE SEARCHES its DATABASE of web pages:

  Page 1: "Top 10 Italian places in MG Road, Bengaluru"   <- KEY matches well!
          Shows you: address, hours, menu, photos          <- this is the VALUE passed to you

  Page 2: "Best French bistro in Mumbai"                   <- KEY does NOT match
          Ignored, gets zero weight in your results.

  Page 3: "Authentic pizza recipes from Naples"            <- KEY partially matches
          Shown but ranked lower (less attention weight).
```

Now apply exactly the same logic to attention in a transformer:

```
In self-attention, EVERY token simultaneously plays THREE roles:

Role 1 — SEARCHER (Query, Q):
  "What information do I need from other tokens to understand my context?"
  Example: the word "sat" asks "who is doing the sitting? what are they sitting on?"

Role 2 — ADVERTISEMENT (Key, K):
  "What kind of information do I contain? How should I present myself to searchers?"
  Example: the word "cat" says "I am a noun, an animal, I could be a subject"

Role 3 — ACTUAL CONTENT (Value, V):
  "What specific information should I pass along when someone attends to me?"
  Example: the word "cat" passes "the specific concept and semantics of 'cat'"

When token A's Query (Q_A) matches token B's Key (K_B):
  -> High attention score -> more of V_B flows into A's output representation
  -> A "learns from" B's information
  -> A's output now incorporates B's context
```

---

### How Q, K, V Are Computed From the Same Input

Here is the surprising part: Q, K, and V all come from the SAME input sequence, just projected
through three different learned weight matrices:

```
+-------------------------------------------------------------------+
|                    Q, K, V COMPUTATION                             |
|                                                                     |
|   Input X (all word embeddings stacked)                            |
|        |                                                           |
|        |                                                           |
|        +---- multiply by W_Q (learned) ---> Q                     |
|        |     "What am I searching for?"                            |
|        |     Each token generates a vector that says:              |
|        |     "Here is the kind of information I need"              |
|        |                                                           |
|        +---- multiply by W_K (learned) ---> K                     |
|        |     "What do I advertise myself as containing?"           |
|        |     Each token generates a vector that says:              |
|        |     "Here is what kind of information I provide"          |
|        |                                                           |
|        +---- multiply by W_V (learned) ---> V                     |
|              "What is my actual content?"                          |
|              Each token generates a vector of actual data          |
|              that gets passed along when attended to               |
+-------------------------------------------------------------------+

CRUCIAL INSIGHT:
W_Q, W_K, and W_V are THREE SEPARATE learned weight matrices.
The SAME input word "cat" produces THREE COMPLETELY DIFFERENT vectors:

  cat -> Q_cat: "I am looking for my verb (what action involves me?) and adjectives"
  cat -> K_cat: "I am a noun, a living creature, a potential subject of a verb"
  cat -> V_cat: "Here is detailed semantic information about the concept of cat"

This separation is what gives attention its incredible expressive power.
The model learns to use Q, K, V in different roles through training.
```

---

### The Full Attention Formula — Every Step Explained

```
Attention(Q, K, V) = softmax( Q x K^T / sqrt(d_k) ) x V

Let's trace each operation with simple numbers.
We have 3 tokens ["dog", "bit", "man"], d_k = 2 (tiny for illustration):

After projecting through W_Q, W_K, W_V (imagined values):

  Q (what each word is searching for):
    dog: [1.0, 0.5]
    bit: [0.0, 1.0]
    man: [0.5, 1.0]

  K (what each word advertises it contains):
    dog: [1.0, 0.0]
    bit: [0.5, 0.5]
    man: [0.0, 1.0]

  V (actual content each word provides):
    dog: [0.9, 0.1]
    bit: [0.1, 0.8]
    man: [0.2, 0.9]

STEP 1: Q x K^T  (compute raw similarity scores)
        This is a dot product between every pair of (query, key):
                 dog    bit    man
  dog's Q:    [ 1.0,  0.75,  0.50 ]  <- dog mostly matches dog's own key
  bit's Q:    [ 0.0,  0.50,  1.00 ]  <- bit matches man's key most!
  man's Q:    [ 0.5,  0.75,  1.00 ]  <- man broadly matches bit and man

STEP 2: Divide by sqrt(d_k) = sqrt(2) = 1.414
        WHY? With larger d_k (like 64), dot products grow very large.
        Large values push softmax into regions where gradients vanish.
        Dividing keeps everything in a healthy range for softmax.
        
                 dog    bit    man
  dog:        [ 0.71,  0.53,  0.35 ]
  bit:        [ 0.00,  0.35,  0.71 ]
  man:        [ 0.35,  0.53,  0.71 ]

STEP 3: Apply softmax to each row (turns scores into probabilities)
        Each row now represents: "how much attention should I pay to each token?"
        
                 dog    bit    man
  dog:        [ 0.42,  0.36,  0.22 ]  <- dog attends mostly to itself
  bit:        [ 0.18,  0.28,  0.54 ]  <- bit attends MOST to man! (the object)
  man:        [ 0.25,  0.33,  0.42 ]  <- man spreads attention around

STEP 4: Multiply by V (compute weighted average of value vectors)

  dog_output = 0.42 * [0.9, 0.1]    <- 42% of dog's own info
             + 0.36 * [0.1, 0.8]    <- 36% of bit's info
             + 0.22 * [0.2, 0.9]    <- 22% of man's info
             = [0.42, 0.68]         <- dog's context-enriched representation

  Notice: "bit" ended up attending most strongly to "man".
  A verb naturally focuses on its object! The model learned this
  alignment from data, without being explicitly programmed.
```

---

## 4. Cross-Attention

###  The Bridge Between Two Sequences

Cross-attention is the mechanism that allows the decoder to "read" from the encoder. The key
difference from self-attention is WHERE Q, K, and V come from:

```
SELF-ATTENTION (everything from same sequence):
  Q, K, V all come from the same input sequence.
  Each word in the sequence talks to every other word in THAT SAME sequence.
  Used in: encoder, and decoder's first attention layer (masked).

CROSS-ATTENTION (two different sequences meet):
  Q comes from the DECODER  <- "What do I (the decoder) need to generate the next word?"
  K comes from the ENCODER  <- "What does the encoder output contain?"
  V comes from the ENCODER  <- "What actual encoder content should flow to the decoder?"

  The decoder uses its own state to search the encoder's memory.
  Think of it as: the writer (decoder) consulting the reader's notes (encoder).
```

---

###  A Real Translation Example

```
Translating English to French.
Encoder has processed: "The cat is on the mat"
Decoder is generating: "Le chat est sur le tapis"

Cross-attention weights at each decoding step:

         |<--- Encoder tokens (English source) --->|
         The    cat    is    on    the    mat
        +----------------------------------------+
"Le"    | 0.40   0.15  0.10  0.10  0.20   0.05  |  "Le" (The) looks at "The" (article)
"chat"  | 0.05   0.82  0.05  0.03  0.03   0.02  |  "chat" (cat) STRONGLY looks at "cat"!
"est"   | 0.05   0.05  0.80  0.05  0.03   0.02  |  "est" (is) STRONGLY looks at "is"!
"sur"   | 0.05   0.05  0.05  0.75  0.05   0.05  |  "sur" (on) STRONGLY looks at "on"!
        +----------------------------------------+

The model has AUTOMATICALLY learned word alignment between languages!
Nobody told it that "chat" corresponds to "cat" -- it discovered this
relationship because learning the alignment helps minimize translation loss.
This is the mechanism behind every major translation system today.
```

---

###  When Cross-Attention Appears

```
USES OF CROSS-ATTENTION:

1. Machine Translation (T5, original Transformer)
   Encoder: reads "The cat sat on the mat" (English)
   Decoder: generates "Le chat etait assis sur le tapis" (French)
   Cross-Attn: each French word attends to relevant English words

2. Text Summarization (BART, Pegasus)
   Encoder: reads 1000-word news article
   Decoder: generates 50-word summary
   Cross-Attn: each summary word attends to the most relevant article parts

3. Speech-to-Text (Whisper)
   Encoder: processes audio mel-spectrogram (the sound waveform)
   Decoder: generates text transcript
   Cross-Attn: each text word attends to the corresponding audio segment

4. Image Captioning (BLIP, Flamingo)
   Encoder: processes image patches (visual features)
   Decoder: generates text description of the image
   Cross-Attn: each word attends to relevant image regions

NOTE FOR MODERN LLMs:
GPT-4, Claude, Llama, Gemini are DECODER-ONLY, so they do NOT use cross-attention.
They only have self-attention (causal/masked). Cross-attention is specifically
the bridge in encoder-decoder architectures.
```

---

## 5. Feed-Forward Neural Networks

### The "Processing" Layer — What Happens After Attention?

Attention is brilliant at figuring out WHICH information to combine from WHERE. But there is a
fundamental limitation: attention is essentially a weighted average. It mixes token representations
together but does not compute complex non-linear functions of the result.

Think of it this way:
- Attention is like **gathering information** — collecting relevant facts from across the sentence
- FFN is like **thinking about that information** — processing those facts into useful conclusions

The FFN is applied to each token INDEPENDENTLY (using the same weights for every position),
acting like a shared "reasoning module" that all tokens pass through one at a time.

---

###  The Expand-Then-Compress Pattern

```
WHY DOES THE FFN EXPAND TO 4x BEFORE COMPRESSING BACK?

Imagine solving a complex math problem:
  Sometimes you need to write out many intermediate steps on a whiteboard
  before you can arrive at the final compact answer.

The FFN does something mathematically similar:
  Expand:  768 -> 3072  "Let me consider 3072 possible angles of this problem"
  Activate: keep positive, filter negative  "These angles are relevant; those aren't"
  Compress: 3072 -> 768  "Here's my summary conclusion in the original 768-dim space"

+-------------------------------------------------------------------+
|                       FFN ARCHITECTURE                             |
|                                                                     |
|  Input from Attention Layer                                        |
|  (batch, seq_len, 768)                                             |
|             |                                                       |
|             v                                                       |
|  +----------------------------+                                    |
|  |  W1: Linear Expansion      |   768 -> 3072 dimensions           |
|  |  bias b1 added             |   Shape of W1: (768, 3072)         |
|  |  Parameters: 768 x 3072   |   = 2,359,296 parameters           |
|  +----------------------------+                                    |
|             |                                                       |
|             v                                                       |
|  +----------------------------+                                    |
|  |  Activation Function       |   Non-linear transformation        |
|  |  (ReLU or GELU or SwiGLU) |   Introduces the non-linearity     |
|  |                            |   that makes deep learning work    |
|  +----------------------------+                                    |
|             |                                                       |
|             v                                                       |
|  +----------------------------+                                    |
|  |  W2: Linear Compression    |   3072 -> 768 dimensions           |
|  |  bias b2 added             |   Shape of W2: (3072, 768)         |
|  |  Parameters: 3072 x 768   |   = 2,359,296 parameters           |
|  +----------------------------+                                    |
|             |                                                       |
|             v                                                       |
|  Output: (batch, seq_len, 768)   <- SAME shape as input!           |
+-------------------------------------------------------------------+
```

---

###  Activation Functions: Why Non-Linearity Is Essential

Without an activation function, you could stack 1000 linear layers and they would all collapse
into a single linear transformation. Non-linearity is what gives neural networks their expressive
power — the ability to learn curved, complex patterns in data.

```
RELU (Rectified Linear Unit — 2012 era, now classic):
  Formula: f(x) = max(0, x)
  
  Example outputs:
    f(-3.0) =  0.0  (clamped to zero)
    f(-0.5) =  0.0  (clamped to zero)
    f( 0.0) =  0.0
    f( 1.5) =  1.5  (unchanged)
    f( 4.0) =  4.0  (unchanged)
  
  Shape:           /
                 /
               /
  ____________/
  
  Simple and fast. Problem: "dying ReLU" — neurons stuck at 0 cannot recover
  because the gradient is exactly 0 for negative inputs.

---------------------------------------------------------------------

GELU (Gaussian Error Linear Unit — 2016, used in BERT, GPT-2, GPT-3):
  Formula: f(x) = x * CDF_of_normal(x)
  
  Example outputs:
    f(-3.0) = -0.004 (slightly negative, not hard zero)
    f(-1.0) = -0.159 (small negative)
    f( 0.0) =  0.0
    f( 1.0) =  0.841
    f( 3.0) =  2.996
  
  Shape:       ___/
           ___/
  ________/
    (small dip below zero near -1)
  
  Smoother than ReLU. The slight negative values for near-zero inputs
  improve gradient flow and model performance. Empirically better than ReLU.

---------------------------------------------------------------------

SWIGLU (2020, used in Llama 2/3, Mistral, Gemma, most modern LLMs):
  Formula: SwiGLU(x) = SiLU(x * W1) * (x * W3)
  
  This is a GATED activation — two separate projections are multiplied:
  
  Branch 1 (Gate):   x -> W1 -> SiLU activation  "How much should flow through?"
  Branch 2 (Up):     x -> W3 -> (no activation)  "What is the content?"
  Combined:          gate * up                    "Selective information flow"
  
  Then compressed back:   combined -> W2 -> output
  
  Result: Three weight matrices (W1, W2, W3) instead of two.
  Cost: More parameters, but researchers reduce d_ff to (2/3)*4*d_model to compensate.
  Benefit: Consistently outperforms GELU on large models empirically.
  Why: The gating mechanism allows the model to selectively suppress or amplify
       different dimensions of the representation — more expressive than a simple activation.
```

---

###  Parameter Count Breakdown — The FFN Has MORE Than Attention

This surprises many people. Let's count precisely for BERT-base:

```
Configuration: d_model=768, n_heads=12, d_ff=3072

ATTENTION LAYER:
  W_Q: 768 x 768 = 589,824 parameters
  W_K: 768 x 768 = 589,824 parameters
  W_V: 768 x 768 = 589,824 parameters
  W_O: 768 x 768 = 589,824 parameters
  TOTAL ATTENTION: ~2,359,296 parameters (~2.4 million)

FFN LAYER:
  W1 (expand):   768 x 3072  = 2,359,296 parameters
  b1 (bias):     3072         =     3,072 parameters
  W2 (compress): 3072 x 768  = 2,359,296 parameters
  b2 (bias):     768          =       768 parameters
  TOTAL FFN: ~4,722,432 parameters (~4.7 million)

Conclusion: FFN has TWICE as many parameters as attention!

Research interpretation:
  - Attention is like "routing" — deciding which information to combine
  - FFN is like "storage" — storing the actual learned knowledge
  - Studies show that factual knowledge ("Paris is the capital of France")
    tends to be stored in FFN weights, while relational reasoning
    (who did what to whom) happens primarily in attention heads.
  - This is why "editing" a model's factual knowledge often targets FFN weights.
```

---

## 6. Layer Normalization

###  The Problem: Training Instability in Deep Networks

Imagine you are training a 96-layer network (like GPT-3). After each layer, numbers get
transformed. After many layers, these transformations compound — numbers can become astronomically
large (activations "explode") or vanishingly small (activations "vanish"). Either extreme makes
training unstable:

```
WHAT CAN GO WRONG WITHOUT NORMALIZATION:

Layer 1 output: values around [ 0.5, 1.2, 0.8, ... ]  <- normal range
Layer 2 output: values around [ 1.1, 2.4, 1.6, ... ]  <- growing slightly
Layer 5 output: values around [ 5.0, 12.0, 8.0, ... ] <- getting large
Layer 10 output: values around [ 50, 120, 80, ... ]    <- much too large
Layer 20 output: values around [ 50000, 120000, ... ]  <- EXPLODED
  -> Softmax of these huge numbers = [1, 0, 0, ...] or [0, 1, 0, ...]
  -> Gradients become zero or infinity
  -> Training crashes or diverges

OR:
Layer 1 output: values around [ 1.0, 2.0, 1.5, ... ]   <- normal range
Layer 10 output: values around [ 0.01, 0.02, 0.015, ...] <- shrinking
Layer 30 output: values around [ 0.0001, 0.0002, ... ]   <- nearly zero
  -> Model's activations carry almost no information
  -> Gradients also vanish
  -> Early layers stop learning
```

Layer Normalization prevents this by standardizing each token's features at every layer.

---

###  What Layer Normalization Does — Every Step

```
Example: one token's feature vector (d_model = 6, for simplicity)

BEFORE LayerNorm:
  Features: [ 12.4,   2.1,   20.8,   0.5,   8.3,   15.6 ]
            <- huge spread! mean=9.95, std=7.4

STEP 1: Compute the mean across all 6 features
  mean = (12.4 + 2.1 + 20.8 + 0.5 + 8.3 + 15.6) / 6
       = 59.7 / 6
       = 9.95

STEP 2: Compute the variance (average squared distance from mean)
  variance = mean of [ (12.4-9.95)^2, (2.1-9.95)^2, (20.8-9.95)^2, ... ]
           = mean of [ 6.0025, 61.6225, 117.7225, 88.2025, 2.7225, 31.6225 ]
           = 51.32

STEP 3: Normalize (center to mean 0, scale to std 1)
  std = sqrt(variance + epsilon) = sqrt(51.32 + 0.000001) = 7.16
  normalized_i = (feature_i - mean) / std

  [ (12.4-9.95)/7.16,  (2.1-9.95)/7.16,  (20.8-9.95)/7.16, ... ]
= [ +0.34,             -1.10,             +1.52,  -1.32,  -0.23,  +0.79 ]

STEP 4: Scale and shift with learned parameters gamma (g) and beta (b)
  output = gamma * normalized + beta
  (gamma initialized to 1, beta initialized to 0; both learned during training)
  
  This final step lets the model "undo" the normalization if that's what's best!
  The model can learn to re-scale to any distribution it finds useful.

AFTER LayerNorm:
  Features: [ +0.34, -1.10, +1.52, -1.32, -0.23, +0.79 ]
            <- controlled scale, mean≈0, std≈1, stable!

Formula: LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

---

###  LayerNorm vs BatchNorm — Why LLMs Use LayerNorm

```
BATCHNORM normalizes across the BATCH dimension:
  For each feature dimension, it computes mean and std across all examples
  in the batch, then normalizes.
  
  Batch of 4 sentences, looking at feature #3 of token #5:
    Sentence 1: feature value = 2.1
    Sentence 2: feature value = 8.4    <- compute mean (4.9) and std (2.5)
    Sentence 3: feature value = 3.7       using THESE 4 values
    Sentence 4: feature value = 5.2
  
  PROBLEMS for NLP:
  1. Variable sequence lengths require padding -> padding distorts batch stats
  2. At inference time, batch size = 1 (generating one response at a time)
     -> cannot compute meaningful batch statistics
  3. Different sequence positions have fundamentally different statistical properties
     -> normalizing them together doesn't make sense

LAYERNORM normalizes across the FEATURE dimension:
  For each token (each word embedding), it computes mean and std across all
  feature dimensions of THAT token, then normalizes independently.
  
  Single token's 768 features:
    Feature 1: 12.4
    Feature 2:  2.1     <- compute mean and std across
    Feature 3: 20.8        ALL 768 features of THIS token
    ...
    Feature 768: 3.2
  
  BENEFITS for NLP:
  1. Works identically regardless of sequence length
  2. Works identically for batch size = 1 at inference
  3. Each token normalized independently -> parallelizable
  4. Empirically proven to work very well for transformer training
```

---

###  Pre-Norm vs Post-Norm — A Critical Architectural Choice

The ORIGINAL 2017 Transformer placed LayerNorm AFTER attention and FFN (post-norm). Modern LLMs
place it BEFORE (pre-norm). This seemingly small change dramatically affects training stability:

```
POST-NORM (Original "Attention Is All You Need", 2017):

  x ──> [Multi-Head Attention] ──> [x + attention_output] ──> [LayerNorm] ──> ...
        "attend first, normalize second"

  Training issue:
    At the start of training, attention outputs can be large and noisy.
    Adding the noisy output to x BEFORE normalization amplifies instability.
    This is why the original paper needed careful learning rate warmup schedules.
    Without warmup, training frequently crashed.

PRE-NORM (GPT-2, GPT-3, Llama, Claude, most models from 2019 onward):

  x ──> [LayerNorm] ──> [Multi-Head Attention] ──> [x + attention_output] ──> ...
        "normalize first, attend second"                      ^
                                                              |
                                              residual adds ORIGINAL x (not normalized x)

  Why this is better:
    1. The attention always receives normalized, stable input -> no exploding attention scores
    2. The residual connection adds the ORIGINAL x unchanged -> direct gradient path
    3. Larger learning rates can be used safely -> faster training
    4. Works reliably for 100+ layers without any warmup tricks
    5. More numerically stable throughout the entire training run

GRADIENT FLOW VISUALIZATION:

Post-Norm:   Loss <-- LayerNorm <-- [x + Attn(x)] <-- x
             Gradient MUST PASS THROUGH LayerNorm backward
             LayerNorm can distort or block the gradient

Pre-Norm:    Loss <-- [x + Attn(LayerNorm(x))] <-- x
                           ^                         ^
                      gradient through Attn     DIRECT gradient
                                                  (gradient = 1)
             The direct path has gradient = 1 (no distortion).
             This gradient highway is critical for training deep models.
```

---

###  RMSNorm — The Streamlined Version Used in Modern Models

Llama 2, Llama 3, Mistral, Gemma, PaLM, and most 2023+ models use RMSNorm instead of LayerNorm:

```
STANDARD LAYERNORM:
  Step 1: Compute mean across features    (centering step)
  Step 2: Subtract mean from each feature
  Step 3: Compute variance
  Step 4: Divide by sqrt(variance + eps) (scaling step)
  Step 5: Multiply by learned gamma      (re-scale)
  Step 6: Add learned beta               (re-shift)
  Learnable params: gamma AND beta (2 params per feature)

RMSNORM (Root Mean Square Normalization):
  Step 1: Compute RMS = sqrt(mean(x^2) + eps)   (combined step, no centering!)
  Step 2: Divide by RMS                          (scaling step)
  Step 3: Multiply by learned gamma              (re-scale)
  Learnable params: gamma ONLY (1 param per feature, no beta)

WHY DOES SKIPPING MEAN SUBTRACTION WORK?
  Researchers (Zhang & Sennrich, 2019) found that the re-centering step
  (subtracting the mean) contributes very little to the actual benefit of
  normalization. The re-scaling (dividing by RMS) does the heavy lifting.
  
  Dropping centering:
    -> Saves ~33% of normalization computation
    -> Removes the beta parameter (one less thing to learn per layer)
    -> Achieves equal or better performance empirically
    -> Roughly 10-15% faster wall-clock time in practice
    -> Slightly simpler gradient computation

This is why every major open-source LLM since 2023 has switched to RMSNorm.
```

---

## 7. Residual Connections

###  The Fundamental Challenge: Vanishing Gradients in Deep Networks

Training a neural network uses gradient descent: you compute how much the loss changes if you
nudge each weight slightly (the gradient), then nudge weights in the direction that reduces loss.
This gradient must flow backward through every single layer.

The problem becomes clear when you think about chain rule:

```
Without residuals, gradient at layer 1 in a 96-layer model:

  gradient_layer1 = gradient_final * (dL96/dL95) * (dL95/dL94) * ... * (dL2/dL1)
                                        ^             ^                    ^
                                   96 multiplications, each typically < 1

  If each factor = 0.9:
  gradient_layer1 = gradient_final * 0.9^95 = gradient_final * 0.00007
                                                                 ^
                                                            Essentially ZERO!

  The first layer receives almost no learning signal.
  It barely updates during training.
  This is the vanishing gradient problem.

  If each factor > 1:
  gradient_layer1 = gradient_final * 1.1^95 = gradient_final * 11,648
                                                                 ^
                                                        INFINITY! Exploding gradients.
  Training crashes with NaN losses.
```

Residual connections elegantly solve both problems with one simple mathematical trick.

---

###  How Residual Connections Work — The Simple Idea

```
WITHOUT residual connection:
  Input x ──> [ Layer F(x) ] ──> Output = F(x)
  
  Layer must learn the complete transformation from x to the desired output.
  All gradient flows through F, which can distort or block it.

WITH residual connection (the "skip connection"):
  
  Input x ────────────────────────────────────────────────────┐
            |                                                   |
            v                                                   | (shortcut/skip path)
         [ Layer F(x) ]                                        |
            |                                                   |
            v                                                   v
        F(x) ─────────────────────────────────────────────> [+] ──> Output = x + F(x)

Key insight: Instead of learning the full mapping F(x) = desired_output,
the layer only needs to learn the RESIDUAL (the difference):
  
  F(x) = desired_output - x   <- just the CHANGE needed, not the full transformation

This is much easier! If the layer should not change x much (which is common in
deep layers that have already learned good representations), F(x) can be small
(close to zero) and the output is approximately x. The model naturally learns
to make small, useful adjustments rather than complete transformations.
```

---

###  The Math — Why Residuals Fix Vanishing Gradients

```
Standard layer: output = F(x)
  Gradient: dL/dx = dL/d_output * dF/dx
  If dF/dx is small -> gradient vanishes

Residual layer: output = x + F(x)
  Gradient: dL/dx = dL/d_output * d(x + F(x))/dx
                  = dL/d_output * (1 + dF/dx)
                                   ^
                                   This "1" is the entire key!
  
  Even if dF/dx ≈ 0 (the layer isn't contributing much):
  dL/dx = dL/d_output * (1 + 0) = dL/d_output * 1 = dL/d_output unchanged!
  
  The gradient flows through UNCHANGED along the residual path.
  This is the "gradient highway" — a direct path from the final layer
  all the way back to the first layer with no attenuation.

PRACTICAL RESULT:
  96-layer network WITHOUT residuals: gradient at layer 1 ≈ 0.00007
  96-layer network WITH residuals:    gradient at layer 1 ≈ 0.95
  
  The difference is not incremental — it's the difference between a model
  that cannot train at all and one that trains perfectly.
  
  GPT-3 has 96 layers. Without residuals, it would be untrainable.
  The residual connection is not a nice-to-have — it is fundamental.
```

---

###  Where Residuals Appear in Each Transformer Block

Every transformer block has exactly TWO residual connections — one around attention, one around FFN:

```
One Complete Transformer Block (Pre-Norm, modern style):

  Input: x
    |
    |   <============ FIRST RESIDUAL PATH ============>
    |   |                                              |
    |   +-----> [LayerNorm] ---> [Multi-Head Attention] ----> [+] ----> x1
    |                                                           ^
    +----------------------------------------------------------+
    
    x1 = x + MultiHeadAttention(LayerNorm(x))
    (the original x is added back, unchanged, to the attention output)
    
    |
    |   <============ SECOND RESIDUAL PATH ============>
    |   |                                               |
    |   +-----> [LayerNorm] ---> [Feed-Forward Net] ----> [+] ----> x2
    |                                                      ^
    +-----------------------------------------------------+
    
    x2 = x1 + FFN(LayerNorm(x1))
    (x1 is added back, unchanged, to the FFN output)
    
  Output: x2 (feeds directly into the next block as its input)

This block is repeated N times:
  GPT-2 Small:  N = 12  blocks
  GPT-3:        N = 96  blocks
  GPT-4:        N = 120 blocks (estimated)
  Llama 3 70B:  N = 80  blocks
  Llama 3 405B: N = 126 blocks

Each block can only make small adjustments thanks to residual connections.
The total transformation is the sum of all N blocks' contributions.
```

---

## 8. Logits and Output

###  The Final Step:

After all the transformer blocks process the input, we have a rich hidden representation for each
token. But we need to convert this back into actual word predictions. This is done through the
"language model head" — a single linear layer that projects from the model's internal dimension
to the vocabulary size.

```
Complete pipeline for predicting the next word:

Prompt: "The quick brown fox"

  Input tokens: [464, 2068, 7586, 21831]  <- token IDs from tokenizer
       |
       v
  Token Embeddings + Positional Encodings
  Shape: (1, 4, 768)  <- 1 batch, 4 tokens, 768-dim vectors
       |
       v  (pass through N transformer blocks)
  Final Hidden State
  Shape: (1, 4, 768)
       |
       | <- we only need the LAST token's representation
       |    (it has "seen" all 4 tokens due to causal attention)
       v
  Last Token State: (1, 768)  <- a single 768-dimensional vector
       |
       | multiply by W_lm_head: (768, 50257)
       v
  LOGITS: (1, 50257)  <- one raw score for EVERY token in the vocabulary
  Values like: [2.1, -0.3, 0.8, ..., 4.2, ..., -1.5]
               <- some tokens have high scores, most have very low scores

  NOTE: Logits are RAW scores. They can be any real number — positive, negative,
  large, small. They are NOT probabilities yet. We need softmax for that.
       |
       v  apply softmax + sampling strategy
  NEXT TOKEN PROBABILITIES
  "jumps"  -> 34.2%   <- model's best guess
  "leaps"  ->  8.1%   <- plausible alternative
  "ran"    ->  6.5%   <- less likely
  "eats"   ->  2.1%   <- unlikely given "brown fox"
  ...50,253 more tokens with probabilities close to 0%
       |
       v  sample one token
  OUTPUT: "jumps"  -> append to prompt -> repeat for next token
```

---

###  Temperature — The Creativity Control Knob

Temperature is a single scalar that controls how "peaked" or "flat" the probability distribution
is. It is the most commonly tuned parameter when deploying LLMs:

```
Formula: P(token_i) = exp(logit_i / T) / sum_j(exp(logit_j / T))

Dividing logits by T before softmax:
  T < 1: makes logits more spread out -> probabilities more extreme -> more confident
  T > 1: makes logits closer together -> probabilities more uniform -> more random
  T = 1: standard softmax, no change

Let's trace with logits = [3.0, 2.0, 1.5, 0.5, -1.0]
for tokens    = ["cat", "dog", "pet", "fox", "car"]

T = 0.3 (very low temperature — deterministic):
  Scaled logits: [10.0, 6.67, 5.0, 1.67, -3.33]
  Probabilities: [94.2%, 3.2%, 1.2%, 0.3%, 0.1%]
  
  "cat" wins overwhelmingly. Almost no diversity. Very predictable.
  Use for: factual Q&A, math problems, code generation where correctness matters.
  
  cat  ######################################################### 94.2%
  dog  ## 3.2%
  pet  # 1.2%

T = 1.0 (standard — balanced):
  Scaled logits: [3.0, 2.0, 1.5, 0.5, -1.0]
  Probabilities: [52.7%, 19.5%, 11.7%, 4.3%, 0.6%]
  
  "cat" leads but alternatives are real possibilities. Natural feel.
  Use for: general chat, most everyday language generation.
  
  cat  ############################ 52.7%
  dog  ########### 19.5%
  pet  ####### 11.7%
  fox  ## 4.3%
  car  . 0.6%

T = 2.0 (high temperature — creative):
  Scaled logits: [1.5, 1.0, 0.75, 0.25, -0.5]
  Probabilities: [33.6%, 20.7%, 16.0%, 9.9%, 5.4%]
  
  Distribution is much flatter. Surprising choices happen regularly.
  Use for: creative writing, brainstorming, poetry, when variety is desired.
  
  cat  ################## 33.6%
  dog  ########### 20.7%
  pet  ######### 16.0%
  fox  ##### 9.9%
  car  ### 5.4%

T -> 0:  Greedy decoding — always picks the single most likely token. Repetitive.
T -> inf: Uniform random — every token equally likely. Meaningless gibberish.
```

---

### Top-k Sampling — Fix the Candidate Pool Size

Even at T=1.0, there are 50,257 tokens in the vocabulary. Many of them are completely irrelevant
(e.g., mathematical symbols, tokens in other languages). Top-k limits sampling to only the k
most likely tokens, preventing the model from "accidentally" choosing absurd options.

```
How Top-k works (k=5 example):

Step 1: Sort all 50,257 tokens by probability (high to low):
  Rank 1:  "cat"   -> 52.7%
  Rank 2:  "dog"   -> 19.5%
  Rank 3:  "pet"   -> 11.7%
  Rank 4:  "fox"   ->  4.3%
  Rank 5:  "bear"  ->  3.1%  <- keep up to here (k=5)
  ──────────────────────────────
  Rank 6:  "bird"  ->  1.5%  <- BLOCKED
  Rank 7:  "fish"  ->  0.9%  <- BLOCKED
  ...
  Rank 50257: some rare token  -> ~0%  <- BLOCKED

Step 2: Re-normalize the kept probabilities to sum to 1
  cat: 52.7 / 91.3 = 57.7%
  dog: 19.5 / 91.3 = 21.4%
  pet: 11.7 / 91.3 = 12.8%
  fox:  4.3 / 91.3 =  4.7%
  bear: 3.1 / 91.3 =  3.4%

Step 3: Sample from these 5 tokens according to re-normalized probabilities.

The problem with fixed top-k:
  When model is VERY CONFIDENT: "Paris" has 98% probability.
  Top-k=50 still includes 49 unlikely tokens that add noise.
  
  When model is VERY UNCERTAIN: all 50,257 tokens have ~0.002% probability.
  Top-k=50 misses 50,207 potentially valid options.
  
  A fixed k is the wrong tool when the model's confidence varies.
```

---

### Top-p (Nucleus) Sampling — The Adaptive Solution

Top-p (also called nucleus sampling) fixes the problem of fixed k by adapting the candidate set
size based on how confident the model actually is:

```
Top-p = 0.9 means:
"Starting from the most likely token, keep adding tokens until their
 combined probability reaches 90%. Use only these tokens for sampling."

SCENARIO 1: Model is very confident (capital city question)
  Sorted probabilities:
    "Paris"  : 88.0%   cumulative: 88.0%   < 90%, keep adding
    "London" :  7.0%   cumulative: 95.0%   >= 90%, STOP HERE
    "Berlin" :  3.0%   <- excluded
    "Rome"   :  1.0%   <- excluded
    ...rest  :  1.0%   <- excluded
  
  Nucleus size: 2 tokens  (model is confident, small nucleus is appropriate)
  Result: almost certainly "Paris", small chance of "London"

SCENARIO 2: Model is uncertain (continuation of a story, many valid options)
  Sorted probabilities:
    "walked" : 18.0%   cumulative: 18.0%
    "ran"    : 15.0%   cumulative: 33.0%
    "sat"    : 13.0%   cumulative: 46.0%
    "stood"  : 12.0%   cumulative: 58.0%
    "moved"  : 10.0%   cumulative: 68.0%
    "turned" :  8.0%   cumulative: 76.0%
    "looked" :  7.0%   cumulative: 83.0%
    "smiled" :  5.0%   cumulative: 88.0%
    "stopped":  3.0%   cumulative: 91.0%   >= 90%, STOP HERE
    ...rest  : 9.0%   <- excluded
  
  Nucleus size: 9 tokens  (model is uncertain, large nucleus allows diversity)
  Result: diverse, creative continuation of the story

TOP-P IS ADAPTIVE:
  - When the model knows the answer: small nucleus (2-3 tokens), confident output
  - When the model is genuinely uncertain: large nucleus (10-50 tokens), diverse output
  - This is why top-p generally outperforms top-k in practice

Typical production settings:
  Factual assistants:  temperature=0.2, top-p=0.9   (mostly deterministic)
  General chatbots:    temperature=0.8, top-p=0.95  (natural, slightly varied)
  Creative writing:    temperature=1.2, top-p=0.95  (genuinely creative)
  Code generation:     temperature=0.2, top-k=10    (must be syntactically correct)
```

---

## Week 3 Summary

###  All Components Working Together — The Full Picture

```
PROMPT: "The cat sat on the"

+------------------------------------------------------------------+
|  STEP 1: TOKENIZE                                                 |
|  "The cat sat on the" -> [464, 3797, 3332, 319, 262]            |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|  STEP 2: EMBED + POSITION ENCODE                                  |
|  Each token ID -> 768-dimensional vector                          |
|  + add positional encoding (so model knows word order)           |
|  Shape: (1, 5, 768)                                               |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|  STEP 3: N TRANSFORMER BLOCKS (repeat N times)                    |
|                                                                   |
|  Each block contains:                                             |
|                                                                   |
|  [RMSNorm] -> [Multi-Head Self-Attention] -> [+ residual]        |
|                     All 5 tokens attend to each other             |
|                     Each of 12 heads captures different patterns  |
|                     Q, K, V matrices do the searching             |
|                                                                   |
|  [RMSNorm] -> [SwiGLU Feed-Forward Network] -> [+ residual]      |
|                     Each token processed independently            |
|                     768 -> 3072 -> 768 dimensions                 |
|                     Applies non-linear reasoning                  |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|  STEP 4: FINAL LINEAR PROJECTION (Language Model Head)            |
|  Last token's hidden state: (1, 768)                              |
|  Multiply by W_lm_head (768, 50257): -> Logits (1, 50257)        |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|  STEP 5: SAMPLING                                                 |
|  Apply temperature (e.g., T=0.8)                                  |
|  Apply top-p filtering (e.g., p=0.95)                             |
|  Sample one token from resulting distribution                     |
+------------------------------------------------------------------+
                               |
                               v
         Next token: "mat" -> append to prompt -> repeat
```

---

###  Complete Concept Reference Table

| Concept | One-Line Explanation | Deep Reason It Exists |
|---------|---------------------|----------------------|
| **Transformer** | Architecture that reads all tokens at once using attention | Faster + better than RNNs at every scale |
| **Multi-Head Attention** | Multiple attention processes in parallel, each specializing | Language has many simultaneous relationship types |
| **Self-Attention** | Every token in a sequence attends to every other token | Captures context that word-by-word reading misses |
| **Q (Query)** | "What information am I searching for?" vector | Each token knows what it needs to understand its context |
| **K (Key)** | "What information do I advertise having?" vector | Each token tells others what type of info it provides |
| **V (Value)** | "What is my actual content?" vector | The information that actually flows when attention is paid |
| **Scale by sqrt(d_k)** | Divide attention scores by square root of head dimension | Prevents large dot products from making softmax too sharp |
| **Cross-Attention** | Q from decoder, K and V from encoder | Bridges source and target sequences in encoder-decoder models |
| **Feed-Forward Network** | Two-layer MLP applied per-token after attention | Provides non-linear reasoning; stores factual knowledge |
| **4x Expansion** | FFN hidden dim = 4 * d_model | More parameters in the middle = more modeling capacity |
| **SwiGLU** | Gated activation: SiLU(xW1) * (xW3) then compress | Empirically outperforms ReLU and GELU at large scale |
| **Layer Normalization** | Normalize each token's features to mean≈0, std≈1 | Stabilizes training, prevents exploding/vanishing activations |
| **Pre-Norm** | Apply norm BEFORE attention/FFN | Better gradient flow, no warmup needed, critical for depth |
| **RMSNorm** | Simplified norm: skip mean subtraction, no beta param | ~15% faster, equal quality; used in all modern open LLMs |
| **Residual Connection** | Output = x + F(x) instead of just F(x) | Gradient highway — makes 100+ layer networks trainable |
| **Logits** | Raw unnormalized scores before softmax | Model's "raw opinion" about each possible next token |
| **Temperature** | Divide logits by T before softmax | Controls distribution sharpness (creativity vs certainty) |
| **Top-p Sampling** | Keep smallest set of tokens with cumulative prob >= p | Adapts candidate pool to model confidence — better than top-k |
| **Top-k Sampling** | Keep only the k most likely tokens | Simple alternative to top-p; limits absurd token choices |

---

## Exercises


**Exercise 1: Implement a complete Transformer encoder block (multi-head attention + FFN + LayerNorm + residual connections) in PyTorch.** 

---

See implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\transformer_architecture_internals\transformer_encoder_block.ipynb`


**Transformer Encoder Block (PyTorch)**

Simple and clean implementation of a **Transformer Encoder Block** using PyTorch.


**What is inside**

This block contains:

* Multi-Head Self Attention
* Feed Forward Network (FFN)
* Layer Normalization
* Residual (skip) connections



**Idea (in simple words)**

Each word in a sentence:

1. Looks at other words (attention)
2. Updates itself
3. Passes through a small neural network



**Flow**

```
Input
  ↓
Multi-Head Attention
  ↓
Add + LayerNorm
  ↓
Feed Forward Network
  ↓
Add + LayerNorm
  ↓
Output
```


**Key Points**

* Input and output shapes are the same
* Attention helps words understand context
* FFN processes each word independently
* Residual connections help training stability

**One-line summary**

**Transformer Encoder = Attention + Small Neural Network + Skip Connections**


**Parameters**

* `d_model`: embedding size
* `num_heads`: number of attention heads
* `d_ff`: hidden size in FFN
* `dropout`: regularization


**Use cases**

* NLP tasks (text classification, translation)
* Feature extraction from sequences
* Building full Transformer models


**Note**

Shape remains same because this block only transforms information, not dimensions.

---

**Exercise 2: Experiment with different numbers of attention heads. How does model performance change?**

---

See the implementation in `C:\Users\SOUMILI\Documents\Code\ai-engineering-learning\transformer_architecture_internals\attention_heads_different.ipynb`

**Transformer Attention Heads Experiment**

**Overview**

This project demonstrates how the **number of attention heads** affects the performance of a Transformer encoder.

We train a simple Transformer model using different numbers of attention heads and compare their loss.


**What This Code Does**

- Creates a dummy dataset (random data)
- Builds a simple Transformer encoder block:
  - Multi-head attention
  - Feedforward network (FFN)
  - Layer normalization
  - Residual connections
- Trains the model with different attention heads:
  - 1, 2, 4, 8
- Prints loss during training
- Plots loss curves
- Compares final performance

**Experiment Setup**

- Sequence length: `10`
- Model dimension (`d_model`): `32`
- Samples: `500`
- Epochs: `20`
- Loss function: `MSELoss`
- Optimizer: `Adam`


**Results**

Example output:

* 1 heads → Final Loss: 1.3506

* 2 heads → Final Loss: 1.3389

* 4 heads → Final Loss: 1.3314

* 8 heads → Final Loss: 1.3437



**Key Observations**

- Increasing heads improves performance up to a point
- **4 heads performed best** in this setup
- **8 heads did not improve further**


**Why This Happens**

- Each head gets part of the model dimension:
   
   head_dim = d_model / num_heads


- More heads → smaller dimension per head
- Too many heads → each head learns less information


**Important Note**

- This experiment uses **random data**
- Results may vary on real datasets
- In real-world models, larger `d_model` allows more heads


**Conclusion**

- More attention heads are helpful, but only up to an optimal point
- Too many heads can reduce performance due to smaller feature size per head


**How to Run**

1. Open Google Colab
2. Run all cells
3. Observe:
 - Training loss
 - Graph
 - Final comparison


**Output**

- Printed loss per epoch
- Loss vs epochs graph
- Final loss comparison table


**Future Improvements**

- Use real NLP dataset
- Add validation accuracy
- Increase model size (`d_model`)
- Visualize attention weights

---

  


**Exercise 3: Implement RMSNorm and compare its speed to standard LayerNorm using PyTorch benchmarking.**

**Exercise 4: Write a function that generates text using temperature + top-p sampling. Compare outputs at different temperatures.**

**Exercise 5: Calculate the total parameters for a Transformer with d_model=512, n_heads=8, d_ff=2048, n_layers=6.**

---




## Further Reading

| Resource | What It Covers |
|----------|---------------|
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | The original Transformer paper | 
| [On Layer Normalization in Transformers (2020)](https://arxiv.org/abs/2002.04745) | Pre-norm vs Post-norm analysis |
| [GLU Variants Improve Transformer (2020)](https://arxiv.org/abs/2002.05202) | SwiGLU paper by Noam Shazeer |
| [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/) | Original paper with line-by-line code |


---

**1. Attention Is All You Need (2017)**

---

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

**2. "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)**

---


**Why this paper**

* Original Transformer used **LayerNorm after each block (Post-LN)**
* This caused:

  * Unstable training 
  * Needs careful tuning (learning rate warmup)
  * Hard to train deep models


**Key Idea**

Two designs:

**1. Post-LN (Old)**

```
Output = LayerNorm(x + Sublayer(x))
```

**2. Pre-LN (New)**

```
Output = x + Sublayer(LayerNorm(x))
```

Difference:

* Post-LN → Normalize AFTER
* Pre-LN → Normalize BEFORE



**Main Finding**

**Pre-LN is much more stable than Post-LN**


**Why Pre-LN works better**

* Better gradient flow 
* Residual connection stays clean 
* No vanishing/exploding gradients 

Easy idea:

* Pre-LN = smooth learning
* Post-LN = unstable learning


**Advantages of Pre-LN**

* Stable training
* Works for deep models
* Faster convergence
* Less tuning needed
* No learning rate warmup required


**Disadvantages of Pre-LN**

* Slightly worse final performance (sometimes)
* Less flexibility in representations


**Pre-LN vs Post-LN**

| Feature        | Pre-LN         | Post-LN         |
| -------------- | -------------- | --------------- |
| Stability      | High           | Low             |
| Deep models    | Works well     | Fails often     |
| Training speed | Fast           | Slow            |
| Warmup needed  | No             | Yes             |
| Final accuracy | Slightly lower | Slightly higher |



**Core Insight**

The real issue is **gradient flow**

* Post-LN → gradients become unstable
* Pre-LN → gradients flow smoothly


**Role of Residual Connections**

* Pre-LN keeps shortcut path clean
* Post-LN disturbs it with normalization

  Clean shortcut = stable training


**Practical Tip**

Use Pre-LN in code:

```python
x = x + sublayer(LayerNorm(x))
```


**Impact**

* Used in modern models (GPT, LLMs)
* Enables training of very deep Transformers
* Became standard practice


**Final Takeaway**

Small change, big impact

```
Move LayerNorm before sublayer
→ Stable + scalable Transformers
```

**One-line Summary**

**Normalize before processing → stable training**

---


