# ML Tool Suite (BM25 RAG, Summarizer, Keyphrases, Language→English, Sentiment)

One project, two faces:

1) **Streamlit app** — a clean, single-page UI to:
   - 📄 **Doc Q&A (RAG-lite)** using BM25/TF-IDF + MMR
   - 📝 **Summarizer** (TextRank)
   - 🏷️ **Keyphrase extraction** (TF-IDF)
   - 🌍 **Detect language → English** (offline MarianMT with online fallback)
   - 💬 **Sentiment** (coarse positive/negative/neutral) using a **hybrid ensemble**
     - ML classifier (calibrated)
     - VADER (rule-based)
     - AFINN (lexicon)
     - Simple emoji & emotion lexicon boosts
   - (Optional) ✨ **Gemini paraphraser** to polish final text only

2) **MCP Server** — exposes the same tools via the [Model Context Protocol](https://modelcontextprotocol.io/):
   - `summarize(text, n_sentences)`
   - `keyphrases(text, top_k, max_ngram)`
   - `sentiment(text)`
   - `detect_translate(text)`

> The project demonstrates classic IR/NLP techniques, practical ML calibration, controllable translation (offline/online), and an interoperable MCP tool surface.

---

## Quickstart

### 0) Create & activate a virtual environment (recommended)

```bash
python -m venv .mcp
source .mcp/bin/activate           # Windows: .mcp\Scripts\activate
python -m pip install --upgrade pip
