# HOW_IT_WORKS — Algorithms, Workflow & Design Choices

This document explains **what each tool does**, **how it works under the hood**, and **why we chose these designs**. It also provides **workflow diagrams** and **MCP call examples**.

---

## High‑Level Workflow

```
User action (UI or MCP)
    ├─ Upload docs (optional) → build BM25/TF‑IDF index
    ├─ Ask Doc Q&A       → retrieve Top‑K chunks → MMR reduce → extractive answer
    ├─ Summarize text    → TextRank → top sentences
    ├─ Keyphrases        → TF‑IDF → top n‑grams
    ├─ Detect → English  → language detect → (offline with timeout) ∨ online → English
    └─ Sentiment         → (optional translate) → normalize →
                           ML head + VADER + AFINN → rule‑based ensemble → label, prob
                                                  ↳ auto‑learn high‑confidence to JSONL
```

---

## Doc Q&A (RAG‑lite with BM25 + MMR)

**Goal:** Answer questions from your uploaded documents without an LLM.

1. **Chunking.** We split documents into overlapping windows (default 800 chars, 120 overlap). Overlap reduces boundary loss.
2. **Indexing.**
   - If available, we use **BM25** (Okapi) — a classic IR scoring that down‑weights very common terms and up‑weights rare ones.
   - Otherwise we fallback to **TF‑IDF cosine similarity**.
3. **Retrieval.** We rank chunks for a query and take the top *K* (controlled by the **Top‑K contexts** slider).
4. **Answer selection (MMR).** We collect sentences from those top chunks and run **Maximal Marginal Relevance** to keep sentences that are both **relevant** and **non‑redundant**. The result is an **extractive answer** (a stitched set of source sentences).
5. **Sources.** We show the top ranked snippets with similarity/score so you can verify provenance.

### Why *Top‑K contexts* matters
- **Low K (e.g., 3–5)** → precise, faster, less chance of noise.
- **High K (e.g., 8–10)** → broader recall, helpful if answers are scattered, but may include irrelevant chunks.
- For short notes, **K=3–5** is usually enough. For long/heterogeneous docs, try **K=5–8**.

---

## Summarizer — TextRank

TextRank builds a **sentence graph** where edges are weighted by TF‑IDF cosine similarity. It then runs **PageRank** to compute importance. We return the top sentences (in original order) for a **faithful extractive summary**. No hallucinations since we never generate text.

**Why:** Simple, fast, and language‑agnostic when documents are in English (we allow language‑normalize via the translation tool if needed).

---

## Keyphrases — TF‑IDF

We compute TF‑IDF over the input, then rank terms/phrases (n‑grams).

- **Top‑K** → how many phrases to return.
- **Max n‑gram** → maximum phrase length (1=single words; 2=two‑word phrases; 3=up to trigrams, etc.).

**Why:** Strong baseline that is easy to interpret; for short inputs we suggest **K=5–10** and **n‑gram≤3**.

---

## Detect Language → English

1. **Detect language** with `langdetect`, returning the top language and probability.
2. **Translate** using a **policy** you choose:
   - **Offline (MarianMT)** — loads a HuggingFace *opus‑mt-xx-en* model per language. First use downloads ~100–400MB.
   - **Auto** — try offline with a timeout; if loading is too slow or fails, we **fallback to online** automatically.
   - **Online only (LibreTranslate)** — calls multiple mirrors and supports both JSON and form payloads.

We return:
- detected language + probability
- English translation
- which engine was used (`marian-mt` or `libretranslate`)
- a clear error if both paths failed

**Why:** This demonstrates **robust engineering**: graceful service degradation + user control.

---

## Sentiment — Ensemble + Auto‑Learning

Pipeline:
1. **Normalize text** (emoji→words, de‑elongation “soooo”→“soo”).
2. **Optional translate to English** (using the same translation policy).
3. **Three heads:**
   - **Your ML model (scikit‑learn)** — calibrated logistic regression over TF‑IDF features trained on labelled data you provide. Exposes probability.
   - **VADER** — rule‑based sentiment tuned for social text; yields a compound score in [-1,1].
   - **AFINN** — lexicon valence; we squash to [-1,1] via tanh for stability.
4. **Decision:**  
   - If **ML prob ≥ 0.70**, trust the ML head.  
   - Else if **VADER conf ≥ 0.60**, trust VADER.  
   - Else if **AFINN conf ≥ 0.55**, trust AFINN.  
   - Else combine them:  
     `S = 0.5*sign(ML)*ML_prob + 0.3*VADER + 0.2*AFINN + emotion_boost`  
     with small boosts for explicit emotion words/emoji/“!!!”.  
     Label by the sign of `S` with a ±0.15 margin; take max magnitude as confidence.
5. **Auto‑learning:** high‑confidence predictions (p≥0.80) append to `data/sentiment_labeled.jsonl`. Periodically retrain to incorporate your domain.

**Why:** This is **ML‑first** and **LLM‑free**, while being robust on real‑world text via ensembling and light self‑training.

---

## MCP — Example Calls

> Shown as pseudo‑JSON. The actual shape depends on the MCP client.

### Build an index from raw text (for Doc Q&A)
```jsonc
{
  "tool": "build_index",
  "args": {
    "texts": ["First document text...", "Second document text..."],
    "chunk_size": 800,
    "overlap": 120
  }
}
```

### Ask a question
```jsonc
{
  "tool": "answer_from_docs",
  "args": { "query": "What is MCP?", "top_k": 5 }
}
```

### Summarize
```jsonc
{ "tool": "textrank_summarize", "args": { "text": "Long paragraph...", "n_sent": 5 } }
```

### Keyphrases
```jsonc
{ "tool": "extract_keyphrases", "args": { "text": "Some text", "top_k": 10, "max_ngram": 3 } }
```

### Detect & Translate
```jsonc
{ "tool": "detect_and_translate", "args": { "text": "नमस्ते नेपाल" } }
```

### Sentiment
```jsonc
{ "tool": "analyze_sentiment", "args": { "text": "I absolutely love this!" } }
```

---

## Design Trade‑offs & Notes

- **Why no embeddings/vector DB?** BM25/TF‑IDF are **fast, reproducible, and dependency‑light**; for small research projects they’re the right level of complexity.
- **Why extractive answers?** 100% grounded — no hallucination risk.
- **Why optional LLM?** To keep the core **learning‑centric** and showcase classical NLP rigor; LLM is used only for surface polish if enabled.
- **Why per‑run translation control?** It teaches users about offline/online trade‑offs and avoids accidental big downloads.

---

## Glossary

- **Top‑K contexts** — number of highest‑scoring chunks retrieved before the answer is assembled. Higher K → more recall, more noise.
- **MMR (Maximal Marginal Relevance)** — balances relevance and diversity to avoid repeating similar sentences.
- **n‑gram** — a phrase of *n* consecutive words; bigger n allows multi‑word keyphrases.
- **Calibration** — mapping model scores to probabilities so thresholds (e.g., 0.70) are meaningful.
- **Pseudo‑labeling** — using confident model predictions as training data to improve the model over time.

---

## Reproducibility Tips

- Fix random seeds when training; use stratified splits.
- Save artifacts to `artifacts/` with versioned filenames.
- Keep a small **hold‑out set** to sanity‑check improvements when auto‑learning is enabled.
