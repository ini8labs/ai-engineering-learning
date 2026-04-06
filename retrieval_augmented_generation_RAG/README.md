# RAG — Retrieval Augmented Generation



---

## The Core Idea 

Imagine hiring a brilliant researcher who has read millions of books — but stopped reading **2 years ago** and sometimes **confuses facts**. That's an LLM without RAG.

Now give that researcher **instant access to a filing cabinet** of your own documents. Before answering any question, they quickly flip through the relevant files and base their answer on what's actually written there. That's **RAG**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THE RAG IDEA                                │
│                                                                     │
│   WITHOUT RAG                       WITH RAG                        │
│   ──────────                        ─────────                       │
│   User asks question                User asks question              │
│         │                                 │                         │
│         ▼                                 ▼                         │
│        AI                          Search your docs              │
│   (relies on memory)                      │                         │
│         │                           "Found 3 relevant chunks"       │
│         ▼                                 │                         │
│    May hallucinate                      ▼                         │
│    Outdated info                      AI + Context              │
│    Can't cite sources                   │                         │
│    No private data                      ▼                         │
│                                      Accurate answer              │
│                                      Fresh information            │
│                                      Cites the source             │
│                                      Uses your private data       │
└─────────────────────────────────────────────────────────────────────┘
```

---

##  Four Problems RAG Solves

### 1 · Hallucination
AI invents facts that sound real but aren't. RAG forces the model to answer **only from retrieved documents** — if the answer isn't in the docs, it says so.

### 2 · Outdated Knowledge
A model trained in 2024 knows nothing about 2025. RAG reads from a **live, up-to-date database** — so answers are always current.

### 3 · Private Data
Your company's reports, policies, and databases are invisible to a generic AI. RAG **connects directly to your own files**.

### 4 · No Citations
Generic AI can't tell you where it got its information. RAG **tracks which document each answer came from** — so you can verify.

---

##  The Full RAG Architecture

RAG has two distinct stages that work together:

```
╔═══════════════════════════════════════════════════════════════════╗
║  STAGE 1 — INDEXING  (run once, or whenever data updates)         ║
╚═══════════════════════════════════════════════════════════════════╝

   Your Documents
  (PDF, Word, HTML, Markdown...)
          │
          ▼
    CHUNKING
  Split documents into small, focused pieces
  "War and Peace" → 2,000 index-card-sized passages
          │
          ▼
   EMBEDDING
  Convert each chunk to a list of numbers (a "vector")
  "The cat sat on the mat" → [0.82, -0.34, 0.15, ...]
          │
          ▼
    VECTOR DATABASE
  Store all the vectors, ready for fast similarity search


╔═══════════════════════════════════════════════════════════════════╗
║  STAGE 2 — QUERYING  (runs on every user question)                ║
╚═══════════════════════════════════════════════════════════════════╝

   User asks: "What's our refund policy?"
          │
          ├─ 1. Convert question to a vector
          │
          ├─ 2. Search vector DB for most similar chunks
          │      → Found: [policy_doc.pdf, terms.html, faq.docx]
          │
          ├─ 3. (Optional) Rerank the results for precision
          │
          ├─ 4. Build the prompt:
          │        System: "Answer only from the provided context"
          │        Context: [3 retrieved chunks pasted here]
          │        Question: "What's our refund policy?"
          │
          └─ 5.  AI generates answer + cites the documents 
```

---

##  RAG vs Fine-tuning — Which One Do You Need

Both improve AI, but in completely different ways:

```
┌──────────────────────────────────────────────────────────────────┐
│                    RAG vs FINE-TUNING                            │
├───────────────────────────┬──────────────────────────────────────┤
│           RAG             │          Fine-tuning                 │
├───────────────────────────┼──────────────────────────────────────┤
│  Update data in seconds │  Retrain = hours + $$              │
│  Auto-cites sources     │  No built-in citations             │
│  Zero training cost     │  Expensive compute                 │
│   Retrieval adds latency │  Fastest inference                 │
├───────────────────────────┼──────────────────────────────────────┤
│  Use for: Q&A, company  │  Use for: Custom tone, format,    │
│    knowledge, live data    │    domain-specific behaviour         │
├───────────────────────────┴──────────────────────────────────────┤
│   BEST RESULTS: Combine both — fine-tune the style,            │
│     use RAG for the knowledge                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

##  Section 2 — Chunking: Splitting Documents into Pieces

Every document must be cut into small, focused chunks before embedding. Think of it like converting a textbook into **index cards** — each card covers exactly one idea.

### Why Not Just Use the Whole Document?
- AI has limited **context window** (memory per call)
- Smaller = **more precise search** results
- You only need the **relevant piece**, not 300 pages

### Chunk Size Sweet Spot

```
                  CHUNK SIZE TRADEOFFS

  SMALL ◄────────────────────────────────► LARGE
  100–300 words    300–800 words    800–2,000 words

   Laser precise      Balanced      Rich context
  Low context         Best default    Less precise

  Use for:            Use for:        Use for:
  Fact lookups        Most cases      Summaries
  FAQ systems         General Q&A     Long-form docs

  ─────────────────────────────────────────────────
  ↩  OVERLAP (10–20% of chunk size)
  Repeat a few sentences between adjacent chunks
  so ideas don't get cut in half at boundaries
```

### 5 Chunking Strategies Compared

```
1. FIXED SIZE — Cut every 500 characters, no matter what
   ─────────────────────────────────────────────────────
   "Hello world, this is a long..." → "Hello world, thi" | "s is a long..."
    Simple, fast, Splits mid-sentence awkwardly
    Best for: Prototyping

2. SENTENCE-BASED — Only cut at sentence endings (. ! ?)
   ───────────────────────────────────────────────────────
   "Hello! This works." → "Hello!" | "This works."
    Clean, natural, Sizes vary wildly
    Best for: General text documents

3. RECURSIVE — Try big separators first, then smaller ones
   ───────────────────────────────────────────────────────
   Priority: paragraphs → sentences → words → characters
    Smart and adaptive, Best general-purpose choice
    Best for: Almost everything (LangChain's default)

4. SEMANTIC — Split where topics actually change (uses AI!)
   ─────────────────────────────────────────────────────────
   Embeds sentences, finds where similarity drops
    Most coherent chunks, Slow (needs embedding model)
    Best for: Documents with clear topic shifts

5. STRUCTURE-AWARE — Follow the document's own headings
   ─────────────────────────────────────────────────────
   # Chapter 1 → one chunk | ## Section A → next chunk
    Preserves document logic, Adds heading metadata
    Best for: Markdown, documentation, technical docs
```

---

##  Section 3 — Embeddings: Words Become Numbers

AI can't understand words directly — it works with **numbers**. An embedding model converts any piece of text into a long list of numbers (called a **vector**) that captures its *meaning*.

### The Magic: Similar Meaning = Similar Numbers

```
  TEXT                     VECTOR (simplified)
  ────────────────────     ──────────────────────────
  "King of England"    →   [0.82, 0.15, -0.34, ...]  ─┐
                                                        ├ very similar!
  "British monarch"    →   [0.79, 0.18, -0.31, ...]  ─┘

  "Python programming" →   [-0.12, 0.67, 0.54, ...]  ─── totally different

  Two texts about the same thing → vectors point in same direction
  → search finds them as "similar" even if words differ completely 
```

### Popular Embedding Models

```
┌──────────────────────────────────────────────────────────────────┐
│                   EMBEDDING MODEL GUIDE                          │
├──────────────────────────┬────────┬─────────────────────────────┤
│ Model                    │  Size  │    When to use    │
├──────────────────────────┼────────┼─────────────────────────────┤
│ OpenAI 3-large (paid)    │  3072  │     Production      │
│ OpenAI 3-small (paid)    │  1536  │     Cost-sensitive  │
│ BGE-large (free)         │  1024  │     Best open option│
│ E5-mistral-7B (free)     │  4096  │     Top open quality│
│ all-MiniLM-L6 (free)     │   384  │     Quick prototypes│
└──────────────────────────┴────────┴─────────────────────────────┘
```

###  Matryoshka Embeddings — Russian Nesting Doll Trick

Some models let you shrink the embedding size **after training**, with no retraining:

```
FULL:  [d1, d2, d3, d4, d5 ... d3072]  → Most accurate,  most storage
       ↓ just take first N dimensions
HALF:  [d1, d2, d3, d4, d5 ... d1024]  → Still great,    3× smaller
MINI:  [d1, d2, d3 ........... d256 ]  → Acceptable,    12× smaller

The first N dimensions always form a valid, working embedding — like
a smaller doll that's still a real doll. You choose the tradeoff at
deployment time, not training time.
```

---

##  Section 4 — Vector Databases: The Search Engine for Meanings

A vector database is purpose-built to store millions of embedding vectors and **find the most similar ones in milliseconds**.

### Popular Choices

```
┌──────────────────────────────────────────────────────────────────┐
│                  VECTOR DATABASE COMPARISON                      │
├────────────┬─────────────────┬───────────────────────────────────┤
│  Database  │   Type          │  Best for                         │
├────────────┼─────────────────┼───────────────────────────────────┤
│ Pinecone   │ Managed cloud   │ Production, zero ops              │
│ Weaviate   │ Open-source     │ Keyword + vector hybrid search    │
│ Qdrant     │ Open-source     │ Fast filtering, Rust performance  │
│ Milvus     │ Open-source     │ Billions of vectors, large scale  │
│ Chroma     │ Open-source     │ Prototyping, small projects       │
│ pgvector   │ PostgreSQL ext. │ Already using Postgres? Start here│
│ FAISS      │ Library         │ Research, full control            │
└────────────┴─────────────────┴───────────────────────────────────┘
```

### How HNSW Makes Search Super Fast

The secret behind fast vector search is the **HNSW algorithm** — like Google Maps on a graph:

```
  LAYER 2 (sparse, long jumps)
  ● ─────────────────── ●

  LAYER 1 (medium density)
  ● ──── ● ──── ● ──── ●

  LAYER 0 (all nodes, dense connections)
  ●─●─●─●─●─●─●─●─●─●─●

  SEARCH:
  1. Start at the top layer → jump to the nearest node
  2. Drop to next layer → jump closer
  3. Repeat until the bottom layer
  4. Return the nearest neighbours found

  Result: O(log N) time — searching 1 billion vectors takes
          roughly the same steps as searching 1 million 
```

### Distance — How "Similar" is Measured

```
  COSINE SIMILARITY (most common for text embeddings)
  ────────────────────────────────────────────────────
  Measures the angle between two vectors

   1.0 = identical meaning    → [0.82, 0.15] vs [0.82, 0.15]
   0.0 = completely unrelated → [1, 0, 0] vs [0, 1, 0]
  -1.0 = opposite meaning     → [1, 0] vs [-1, 0]
```

---

##  Section 5 — Building a Complete RAG Pipeline

### Full Architecture at a Glance

```
  USER ASKS A QUESTION
         │
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 1 · EMBED the query (convert to vector)               │
  └────────────────────────────┬────────────────────────────────┘
                               │
         ┌─────────────────────┴────────────────────┐
         ▼                                          ▼
  ┌────────────────┐                       ┌────────────────────┐
  │  BM25 SEARCH   │                       │  VECTOR SEARCH     │
  │  (exact words) │                       │  (similar meaning) │
  └────────────────┘                       └────────────────────┘
         │                                          │
         └──────────────── merge ──────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 3 · RERANK — cross-encoder picks the best few          │
  └──────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 4 · BUILD PROMPT                                       │
  │  System: "Answer only from the provided context."            │
  │  Context: [Retrieved chunks pasted here]                     │
  │  Question: [User's original question]                        │
  └──────────────────────────────────────────────────────────────┘
                               │
                               ▼
                AI GENERATES ANSWER + CITATIONS 
```

###  Hybrid Search — Best of Both Worlds

Keyword search finds exact words. Semantic search finds meaning. Together they're unstoppable:

```
  Query: "Python AI development"

  BM25 (keyword):           Vector (semantic):
  ─────────────────────     ─────────────────────────────────
  Finds "Python" exactly    Finds "machine learning tools"
  Finds "AI" exactly        Finds "deep learning frameworks"
  Misses synonyms           Finds "neural network library"
                            Misses exact string matches

  Combined score = α × semantic_score + (1-α) × bm25_score
                 α = 0.5 is a strong default

  → Gets BOTH exact matches AND related concepts in one search 
```

###  Reranking — The Precision Booster

Retrieval optimises for **recall** (find everything). Reranking optimises for **precision** (rank correctly):

```
  STAGE 1 — Fast bi-encoder retrieval
  ─────────────────────────────────────
  Query embeds once → compare against all docs
  Retrieves top 20 candidates in ~1ms
  (Good recall, imprecise ordering)

       ↓ top 20 candidates pass through

  STAGE 2 — Precise cross-encoder reranking
  ──────────────────────────────────────────
  For each candidate: AI reads (query + document) together
  Scores each pair with full attention across both texts
  Returns top 3–5 candidates in correct order
  (Small latency cost → huge precision gain)

  Before reranking:         After reranking:
  #1 somewhat relevant  →   #1 MOST relevant 
  #2 very relevant      →   #2 relevant      
  #3 very relevant      →   #3 relevant      
  #4 unrelated          →   (dropped)
  #5 unrelated          →   (dropped)
```

---

##  Section 6 — Advanced RAG Techniques (2025–2026)

Basic RAG is powerful. These techniques make it exceptional:

---

### 1 · Multi-Query RAG — Ask From Multiple Angles

One question from one angle might miss relevant docs. Generate variations:

```
  User asks: "How do neural networks learn?"
             │
             ▼
         LLM generates 3 more versions:
         ├── "What is backpropagation?"
         ├── "Training process for deep learning models"
         └── "How does gradient descent update weights?"

  Search all 4 queries → Deduplicate → Merge results

  Result: Up to 4× better recall than a single query 
```

---

### 2 · RAG Fusion — Smart Ranking Across Multiple Searches

Like Multi-Query, but uses **Reciprocal Rank Fusion (RRF)** to combine rankings intelligently:

```
  Doc A ranks #1 in Search 1,  #2 in Search 2  → Very high RRF score
  Doc B ranks #5 in Search 1, #20 in Search 2  → Low RRF score

  RRF score = 1/(60 + rank_1) + 1/(60 + rank_2) + ...

  Documents that consistently rank well across ALL searches
  float to the very top — robust, trustworthy ordering 
```

---

### 3 · Self-RAG — AI That Decides When to Search

Instead of always retrieving (slow) or never retrieving (risky), Self-RAG teaches the AI to decide:

```
  Question arrives
         │
         ▼
  [Should I retrieve?] ─── No ──→ Answer from memory (fast)
         │ Yes
         ▼
  [Is this doc relevant?] ── No ──→ Skip it, try next
         │ Yes
         ▼
  [Does my answer match the doc?] ── No ──→ Revise the answer
         │ Yes
         ▼
   Verified, grounded answer

  The AI becomes self-correcting — it checks its own work 
```

---

### 4 · GraphRAG (by Microsoft) — AI Understands Connections

Standard RAG retrieves isolated chunks. GraphRAG understands **how ideas connect**:

```
  INDEXING PHASE:
  ─────────────────────────────────────────────────────
  Documents → LLM extracts entities & relationships

  "GPT was trained by OpenAI using transformer architecture"
                 ↓
        GPT ──[trained by]──→ OpenAI
        GPT ──[uses]────────→ Transformer
        Transformer ──[enables]──→ Attention mechanism

  → Builds a full knowledge graph of the entire corpus
  → Groups related concepts into "communities"

  QUERYING PHASE:
  ─────────────────────────────────────────────────────
  "What are the main AI research themes?"
  → Global search: read community summaries (great for big picture)

  "How is BERT related to transformers?"
  → Local search: traverse the knowledge graph (great for specific facts)

  Standard RAG can't answer corpus-level questions. GraphRAG can 
```

---

### 5 · Contextual Retrieval (by Anthropic) — Give Chunks Memory

**The Problem:** A chunk saying *"Revenue grew 15% year-over-year"* is useless without knowing *which company* and *which year*.

**The Fix:** Prepend the broader document context to each chunk before embedding:

```
  BEFORE (ambiguous chunk):
  ──────────────────────────────────────────────────────
  "Revenue grew 15% year-over-year, driven by services."

  ↓  LLM reads the full document and writes a context note

  AFTER (contextualised chunk):
  ──────────────────────────────────────────────────────
  "Context: This is from Apple Inc.'s Q3 2025 earnings
  report, discussing quarterly financial performance.

  Revenue grew 15% year-over-year, driven by services."

  Now each chunk understands itself — search precision improves 
```

---

### 6 · Late Chunking — Embed First, Split Later

Traditional chunking **loses context across chunk boundaries**. Late chunking preserves it:

```
  TRADITIONAL:
  Document → Split into chunks → Embed each chunk independently

  Problem: Each chunk only "knows" its own local content.
  A chunk starting mid-sentence has no idea what came before.

  ──────────────────────────────────────────────────────────

  LATE CHUNKING:
  Document → Run entire document through the embedding model
           → Get one vector per TOKEN (word-piece)
           → THEN split those token vectors into chunks
           → Pool each chunk's token vectors into one vector

  Result: Every chunk's embedding carries context from the
          entire document it came from — much richer search 
```

---

##  Key Takeaways 

```
┌─────────────────────────────────────────────────────────────────────┐
│                      7 GOLDEN RULES OF RAG                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1.  RAG fixes the 4 core LLM failures: hallucination, staleness, │
│        private data blindness, and no citations                     │
│                                                                     │
│  2.   Chunking strategy is the #1 tuning lever — bad chunks       │
│        poison the entire pipeline. Use recursive splitting as       │
│        your safe default                                            │
│                                                                     │
│  3.  Embedding model quality matters — OpenAI 3-large for paid,   │
│        BGE-large for free. Match chunk size to what the model       │
│        was trained on (~256–512 tokens)                             │
│                                                                     │
│  4.   HNSW gives you log-scale search at high recall —           │
│        Chroma or FAISS to start, Qdrant/Weaviate for production     │
│                                                                     │
│  5.  Hybrid search (BM25 + vector) outperforms either alone —     │
│        α = 0.5 is your starting point                               │
│                                                                     │
│  6.  Always rerank — cross-encoder reranking adds precision with   │
│        minimal latency. It's one of the highest-ROI improvements    │
│                                                                     │
│  7.  For complex needs: GraphRAG (corpus-level reasoning),        │
│        Contextual Retrieval (richer chunks), Self-RAG (adaptive)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

##  Quick Setup

```bash
# Core libraries
pip install sentence-transformers chromadb faiss-cpu rank-bm25

# For LangChain-based pipeline
pip install langchain langchain-openai langchain-community

# For PostgreSQL vector search
pip install psycopg2-binary

# Optional: OpenAI embeddings and generation
pip install openai
```

---

**1.  Generate and Compare Embeddings**

---

See implementation in `E:\ini8_labs\ai-engineering-learning\retrieval_augmented_generation_RAG\embedding_compare_response.ipynb`

**Sentence Similarity using Embeddings**

Ever wondered if *"I love ML"* and *"I enjoy deep learning"* mean the same thing to a machine?  
This project does exactly that — it teaches your computer to **feel** the meaning of sentences and compare them!


**What It Does**

-  Takes a bunch of sentences as input
-  Converts them into **embeddings** — smart numerical representations of meaning
-  Measures how similar they are using **cosine similarity**
-  Returns a score: **1.0 = twins**, **0.0 = strangers**


**Get Started**

```bash
pip install sentence-transformers numpy
```


**How It Works**

| Step | What's Happening Under the Hood |
|------|----------------------------------|
|  Load Model | Pulls in `all-MiniLM-L6-v2` — small but mighty |
|  Encode | Each sentence becomes a vector of 384 numbers |
|  Compare | Cosine similarity finds the "angle" between meanings |



**Example Output**

```
Cosine Similarity Matrix:
# Every sentence pair gets a score — the closer to 1, the more alike!

Cosine similarity between:
'I love machine learning.'
and
'I enjoy studying deep learning.'
= 0.7839   Pretty similar.
```


**Quick Snippet**

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

e1 = model.encode("I love AI.")
e2 = model.encode("Machine learning is great.")

score = util.cos_sim(e1, e2)
print(f"Similarity: {score.item():.4f}")  # e.g. 0.8500
```



**Model Used**

**`all-MiniLM-L6-v2`**  
Fast  · Lightweight  · Surprisingly powerful 
Perfect for semantic search, clustering, and similarity tasks.

---
