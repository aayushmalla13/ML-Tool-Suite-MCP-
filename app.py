# app.py
import os, json
from io import BytesIO
from typing import Dict, Any, List, Optional

import streamlit as st

from tools import (
    build_rag_index_from_texts, answer_from_docs,
    textrank_summarize, extract_keyphrases,
    analyze_sentiment, detect_and_translate,
    set_translation_policy,  # translation engine control
)

# --------- Sentiment auto-learning ---------
SENT_PSEUDO_ADD = 0.80
AUTO_TRAIN_SENT_EVERY = 30

# --------- Page setup ---------
st.set_page_config(page_title="ML Tool Suite (BM25 RAG, MCP)", page_icon="🧠", layout="wide")
st.title("🧠 ML Tool Suite (MCP)")
st.caption(
    "RAG-lite (BM25) • TextRank Summarizer • TF-IDF Keyphrases • "
    "Sentiment (auto-learning) • Detect language → English. "
    "Optional Gemini polish (off by default)."
)

# --------- How to use ---------
with st.expander("✅ How to use (simple) & examples", expanded=True):
    st.markdown("""
**One page. One column.** Open any tool below, paste text/question, click **Run**, and your result appears right under that tool.

**Examples**
- **Doc Q&A**: “What is Model Context Protocol?” (upload docs first)
- **Summarize**: Paste a long paragraph → get a short extractive summary
- **Keyphrases**: Paste any text → get top keywords/phrases
- **Detect → English**: “नमस्ते नेपाल” → language + English translation
- **Sentiment**: “I absolutely love this!” → positive (p≈0.9)

**Tips**
- In **Detect → English**, you can override the translation engine for a single run (Auto / Offline / Online).
- In **Sentiment**, you can **mark the correct label** under “Was this correct?” to help the model learn.
""")

# --------- Sidebar: translation policy (default) + optional Gemini polish ---------
with st.sidebar:
    st.header("Translation settings")
    engine = st.radio(
        "Engine (default)",
        ["Auto (try offline, fallback online)", "Offline (download if needed)", "Online only"],
        index=0,
        help=("Offline uses MarianMT models (large first-time download per language). "
              "Auto will try offline with a timeout and fall back to online if it takes too long.")
    )
    timeout = st.number_input(
        "Offline load timeout (sec, for Auto)",
        min_value=5, max_value=120, value=25, step=5,
        help="If the offline model doesn't load within this time, we'll use online translation instead (Auto mode)."
    )

    # Apply global default policy AND remember it so we can restore after per-run overrides
    if engine.startswith("Auto"):
        set_translation_policy("auto", offline_timeout_s=int(timeout))
        st.session_state["policy"] = ("auto", int(timeout))
    elif engine.startswith("Offline"):
        set_translation_policy("offline", offline_timeout_s=0)   # blocking load, may download
        st.session_state["policy"] = ("offline", 0)
    else:
        set_translation_policy("online_only", offline_timeout_s=int(timeout))
        st.session_state["policy"] = ("online_only", int(timeout))

    st.divider()
    st.header("Optional: Enhance wording (Gemini)")
    enhance = st.toggle(
        "Enhance with Gemini",
        value=False,
        help="Only paraphrases outputs for readability; ML still does the actual work."
    )
    gemini_key = st.text_input("Gemini API Key", type="password") if enhance else None
    if enhance:
        st.caption("Used only to paraphrase/clean text. Not required.")

    st.divider()
    if st.button("🧹 Clear outputs"):
        for k in ("rag_index", "qa_out", "summ_out", "keys_out", "lang_out", "sent_out", "sent_count"):
            st.session_state.pop(k, None)
        st.success("Cleared.")

# --------- Optional paraphraser (Gemini only) ---------
def paraphrase(text: str, key: Optional[str]) -> Optional[str]:
    if not (text and key):
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = (
            "Rewrite the following so it is clear, concise, and friendly. "
            "Return only the rewritten text.\n\n" + text
        )
        out = model.generate_content(prompt).text or ""
        return out.strip()
    except Exception:
        return None

# --------- Document uploader for RAG ---------
def _read_file_to_text(up) -> str:
    name = up.name.lower()
    if name.endswith(".txt"):
        return up.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(up.read()))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception:
            return ""
    return ""

with st.container(border=True):
    st.subheader("📄 Upload documents (for Doc Q&A)")
    ups = st.file_uploader("Upload .txt/.pdf (multiple allowed)", type=["txt","pdf"], accept_multiple_files=True)
    if ups:
        texts = []
        for f in ups:
            t = _read_file_to_text(f)
            if t:
                texts.append(t)
        if texts:
            st.session_state["rag_index"] = build_rag_index_from_texts(texts)
            chunks = len(st.session_state["rag_index"]["chunks"])
            st.success(f"Indexed {chunks} chunk(s).")

# =========================
# Tool Shortcuts (expanders)
# =========================

# ---- Doc Q&A (RAG-lite BM25) ----
with st.expander("🔎 Doc Q&A (RAG-lite, BM25)", expanded=True):
    q = st.text_input("Your question", placeholder="e.g., What is MCP?")
    topk = st.slider("Top-K contexts", 1, 10, 5)
    if st.button("Run Q&A"):
        idx = st.session_state.get("rag_index")
        res = answer_from_docs(q or "", idx, top_k=topk)
        if "error" in res:
            st.error(res["error"])
        else:
            ans = (res.get("extractive_answer") or "").strip()
            if enhance and ans:
                better = paraphrase(ans, gemini_key)
                if better:
                    ans = better
            st.session_state["qa_out"] = {"answer": ans, "matches": res.get("matches", [])}

    if st.session_state.get("qa_out"):
        out = st.session_state["qa_out"]
        if out.get("answer"):
            st.markdown("**Answer**")
            st.success(out["answer"])
        with st.expander("Show sources (optional)", expanded=False):
            for m in out.get("matches", []):
                st.markdown(f"- **[{m['rank']}]** (score {m['score']:.3f}) — {m['snippet']}")

# ---- Summarizer ----
with st.expander("📝 Summarizer (TextRank)", expanded=False):
    text = st.text_area("Text to summarize", height=160, placeholder="Paste paragraph(s)…")
    n = st.number_input("Sentences", 1, 12, 5, step=1)
    if st.button("Summarize"):
        sents = textrank_summarize(text or "", n_sent=int(n))
        summary = " ".join(sents) if sents else ""
        if enhance and summary:
            better = paraphrase(summary, gemini_key)
            if better:
                summary = better
        st.session_state["summ_out"] = summary

    if st.session_state.get("summ_out"):
        st.markdown("**Summary**")
        st.info(st.session_state["summ_out"] or "No summary.")

# ---- Keyphrases (clear UI) ----
with st.expander("🏷️ Keyphrases (TF-IDF)", expanded=False):
    st.markdown("Extract the most informative words/phrases using TF-IDF.")
    text = st.text_area(
        "Text",
        height=140,
        placeholder="Paste text to extract keywords and short phrases…",
        key="keys_text",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        top_k = st.number_input(
            "How many phrases (Top-K)",
            min_value=3, max_value=50, value=10, step=1,
            help="We rank phrases by TF-IDF score and return the top K."
        )
    with col_b:
        max_ng = st.select_slider(
            "Max phrase length (n-gram)",
            options=[1, 2, 3, 4, 5],
            value=3,
            help="n-gram = words per phrase. 1=single words, 2=two-word phrases, 3=up to three-word phrases, etc."
        )
    st.caption("Tip: For short texts, try K=5–10 and n-gram ≤ 3.")

    if st.button("Extract"):
        phrases = extract_keyphrases(text or "", top_k=int(top_k), max_ngram=int(max_ng))
        st.session_state["keys_out"] = phrases

    if st.session_state.get("keys_out") is not None:
        st.markdown("**Keyphrases**")
        if st.session_state["keys_out"]:
            st.write(", ".join(st.session_state["keys_out"]))
        else:
            st.info("No phrases found.")

# ---- Detect language → English (with per-run engine) ----
with st.expander("🌍 Detect language → English", expanded=False):
    st.markdown("Detect the input language and translate to English with your chosen engine.")
    # Per-run engine choice (overrides sidebar default ONLY for this click)
    st.markdown("**Engine for this run**")
    run_engine = st.radio(
        "Choose how to translate this specific input",
        ["Use sidebar default", "Auto (try offline, fallback online)", "Offline (download if needed)", "Online only"],
        index=0, horizontal=False, key="lang_run_engine"
    )
    run_timeout = st.number_input(
        "Offline load timeout (sec, for Auto in this run)",
        5, 120, 25, 5, key="lang_run_timeout"
    )
    confirm_offline = False
    if run_engine.startswith("Offline"):
        st.warning("Offline may download a large MarianMT model (~100–400MB) the first time for this language.")
        confirm_offline = st.checkbox("I agree to download if needed for this run", value=False, key="lang_run_confirm")

    text = st.text_area("Text", height=120, placeholder="Paste non-English text…", key="lang_text")

    if st.button("Detect & Translate"):
        # Save current global policy so we can restore after this run
        prev_mode, prev_to = st.session_state.get("policy", ("auto", 25))

        # Apply per-run override
        if run_engine.startswith("Use sidebar"):
            pass  # keep current global policy
        elif run_engine.startswith("Auto"):
            set_translation_policy("auto", offline_timeout_s=int(run_timeout))
        elif run_engine.startswith("Offline"):
            if confirm_offline:
                set_translation_policy("offline", offline_timeout_s=0)  # may download (blocking)
            else:
                set_translation_policy("online_only", offline_timeout_s=int(run_timeout))
        else:
            set_translation_policy("online_only", offline_timeout_s=int(run_timeout))

        # Run
        res = detect_and_translate(text or "")

        # Restore previous global policy
        set_translation_policy(prev_mode, offline_timeout_s=int(prev_to))

        english = res.get("english") or "[translation unavailable]"
        if enhance and res.get("english"):
            better = paraphrase(english, gemini_key)
            if better:
                english = better
        st.session_state["lang_out"] = {
            "language": res.get("language"),
            "prob": res.get("language_prob"),
            "english": english,
            "source": res.get("source"),
            "error": res.get("error"),
        }

    if st.session_state.get("lang_out"):
        o = st.session_state["lang_out"]
        st.markdown("**Detection**")
        prob = o.get("prob")
        prob_str = f" (p={prob:.2f})" if isinstance(prob, (int, float)) else ""
        st.write(f"{o.get('language','?')}{prob_str}")
        st.markdown("**English**")
        if o.get("english") and o["english"] != "[translation unavailable]":
            st.success(o["english"])
        else:
            st.warning(o["english"])
        meta_bits = []
        if o.get("source"): meta_bits.append(f"via {o['source']}")
        if o.get("error"):  meta_bits.append(o["error"])
        if meta_bits: st.caption(" · ".join(meta_bits))

# ---- Sentiment (coarse + feedback to learn) ----
with st.expander("💬 Sentiment (positive • negative • neutral)", expanded=False):
    text = st.text_area("Text", height=120, placeholder="Paste text for sentiment…", key="sent_text")
    if st.button("Analyze"):
        res = analyze_sentiment(text or "")
        st.session_state["sent_out"] = res

        # Auto-learn high-confidence examples
        try:
            prob = float(res.get("prob", 0.0))
            label = res.get("label")
            if prob >= SENT_PSEUDO_ADD and (text or "").strip():
                os.makedirs("data", exist_ok=True)
                with open("data/sentiment_labeled.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"text": text, "label": label, "source": "pseudo", "prob": prob}) + "\n")
                cnt = st.session_state.get("sent_count", 0) + 1
                st.session_state["sent_count"] = cnt
                if cnt >= AUTO_TRAIN_SENT_EVERY:
                    try:
                        from ml.sentiment_model import train_sentiment_model
                        train_sentiment_model()
                        st.session_state["sent_count"] = 0
                        st.toast("Sentiment model retrained from new pseudo-labels.", icon="✅")
                    except Exception:
                        pass
        except Exception:
            pass

    if st.session_state.get("sent_out"):
        o = st.session_state["sent_out"]
        st.markdown("**Sentiment**")
        st.success(f"{o.get('label','?')} (p={o.get('prob',0):.2f})")
        src = o.get("source")
        if src:
            st.caption(f"Decision source: {src}")

        # ---------- NEW: feedback UI to teach the model ----------
        with st.expander("Was this correct? (help it learn)", expanded=False):
            current = o.get("label","neutral")
            labels = ["positive","neutral","negative"]
            try:
                idx = labels.index(current)
            except Exception:
                idx = 1
            gold = st.radio("Correct label", labels, index=idx, horizontal=True, key="sent_gold_choice")
            if st.button("Save correction"):
                os.makedirs("data", exist_ok=True)
                txt = st.session_state.get("sent_text","")
                with open("data/sentiment_labeled.jsonl","a",encoding="utf-8") as f:
                    f.write(json.dumps({"text": txt, "label": gold, "source": "human"}) + "\n")
                st.success("Thanks! This correction will be used in the next retrain.")
