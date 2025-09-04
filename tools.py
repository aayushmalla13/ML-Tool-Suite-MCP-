# tools.py
from __future__ import annotations
import re
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect_langs

# ===== Sentiment: your calibrated ML model =====
from ml.sentiment_model import analyze_sentiment_label_proba

# ===== Optional VADER for stronger polarity cues =====
_VADER = None
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        _VADER = SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
        _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None  # gracefully skip if unavailable


def _vader_label_prob(text: str) -> Optional[Tuple[str, float, float]]:
    """
    Returns (label, confidence, compound). Confidence in [0,1], compound in [-1,1].
    """
    if _VADER is None or not text.strip():
        return None
    s = _VADER.polarity_scores(text)
    c = float(s.get("compound", 0.0))
    if c >= 0.5:
        lab = "positive"
    elif c <= -0.5:
        lab = "negative"
    else:
        lab = "neutral"
    conf = min(1.0, abs(c))
    return lab, conf, c


# ===== Third head: AFINN (lexicon valence) =====
_AFINN = None
try:
    from afinn import Afinn
    _AFINN = Afinn(emoticons=True)
except Exception:
    _AFINN = None


def _afinn_score(text: str) -> Optional[Tuple[str, float, float]]:
    """
    Returns (label, confidence, score_norm) or None.
    Raw AFINN score is unbounded-ish; normalize via tanh(score/5) to [-1,1].
    """
    if _AFINN is None or not text.strip():
        return None
    raw = float(_AFINN.score(text))
    norm = float(np.tanh(raw / 5.0))  # [-1,1]
    if norm >= 0.25:
        lab = "positive"
    elif norm <= -0.25:
        lab = "negative"
    else:
        lab = "neutral"
    conf = min(1.0, abs(norm))
    return lab, conf, norm


# ===== Light pre-processing for sentiment =====
_RE_EXCL = re.compile(r"!+")
_RE_ELONG = re.compile(r"(.)\1{2,}", re.IGNORECASE)  # looove -> loove
_POS_BOOST = {
    "love", "amazing", "great", "awesome", "excellent",
    "happy", "joy", "glad", "delighted", "fantastic"
}
_NEG_BOOST = {
    "hate", "terrible", "awful", "horrible",
    "furious", "angry", "rage", "disgusted",
    "anxious", "anxiety", "panic", "worried"
}
_EMOJI_MAP = {
    "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy", "🙂": "happy", "😊": "happy",
    "😍": "love", "😘": "love",
    "😂": "haha", "🤣": "haha", "😅": "haha",
    "😢": "sad", "😭": "sad", "☹️": "sad", "🙁": "sad",
    "😠": "angry", "😡": "angry", "🤬": "angry",
    "😱": "shock", "😨": "anxious", "😰": "anxious", "😟": "worried", "😞": "sad", "😔": "sad",
}


def _normalize_for_sentiment(t: str) -> str:
    t = (t or "").strip()
    # map emojis to words (helps classical models & lexicons)
    for e, w in _EMOJI_MAP.items():
        t = t.replace(e, f" {w} ")
    # reduce character elongation (soooo → soo)
    t = _RE_ELONG.sub(r"\1\1", t)
    return t


def _emotion_boost(text: str) -> float:
    """
    Simple lexicon bump based on strong emotion words and exclamation marks.
    Range roughly [-0.25, +0.25]
    """
    low = text.lower()
    pos = sum(1 for w in _POS_BOOST if w in low)
    neg = sum(1 for w in _NEG_BOOST if w in low)
    exclam = len(_RE_EXCL.findall(text))
    score = 0.08 * pos - 0.10 * neg + 0.02 * min(exclam, 5)
    # clamp
    return float(max(-0.25, min(0.25, score)))


# ===== MarianMT (optional) with timeout & session backoff =====
_MT_AVAILABLE = True
try:
    from transformers import pipeline  # type: ignore
except Exception:
    _MT_AVAILABLE = False

_MT_CACHE: Dict[str, Any] = {}     # lang -> pipeline
_MT_FAILED: set[str] = set()       # langs that timed out/failed in this session

# Global translation policy (controlled by UI)
# mode: "auto" | "offline" | "online_only"; timeout used in "auto"
_TRANSLATION_POLICY = {"mode": "auto", "timeout_s": 25}

# Popular language → English Marian models
_MARIAN_MODELS = {
    "ne": "Helsinki-NLP/opus-mt-ne-en", "hi": "Helsinki-NLP/opus-mt-hi-en",
    "bn": "Helsinki-NLP/opus-mt-bn-en", "ur": "Helsinki-NLP/opus-mt-ur-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en", "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en", "it": "Helsinki-NLP/opus-mt-it-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en", "ru": "Helsinki-NLP/opus-mt-ru-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en", "fa": "Helsinki-NLP/opus-mt-fa-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en", "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en", "id": "Helsinki-NLP/opus-mt-id-en",
    "tr": "Helsinki-NLP/opus-mt-tr-en", "ta": "Helsinki-NLP/opus-mt-ta-en",
}


def set_translation_policy(mode: str = "auto", offline_timeout_s: int = 25) -> None:
    """
    Configure translation behavior globally. Called from the UI.
    modes:
      - "auto"        : try offline MarianMT with timeout, then fallback to online
      - "offline"     : always attempt MarianMT (may download large models); no timeout
      - "online_only" : always use LibreTranslate/MyMemory (no downloads)
    """
    m = str(mode).lower().strip()
    if m not in {"auto", "offline", "online_only"}:
        m = "auto"
    _TRANSLATION_POLICY["mode"] = m
    _TRANSLATION_POLICY["timeout_s"] = max(5, int(offline_timeout_s or 25))


def _load_marian_with_timeout(model_name: str, timeout_s: Optional[int]):
    """
    Load a HF translation pipeline with a hard timeout (to avoid long stalls).
    If timeout_s is None → blocking load (used in 'offline' mode).
    """
    if timeout_s is None:
        return pipeline("translation", model=model_name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(pipeline, "translation", model=model_name)
        try:
            return fut.result(timeout=timeout_s)
        except Exception:
            try:
                fut.cancel()
            except Exception:
                pass
            return None


def _get_mt_pipeline(src_lang: str):
    if not _MT_AVAILABLE:
        return None
    lang = (src_lang or "").lower()
    if not lang:
        return None
    if lang in _MT_CACHE:
        return _MT_CACHE[lang]
    if lang in _MT_FAILED:
        return None  # don't keep retrying within this session

    model_name = _MARIAN_MODELS.get(lang)
    if not model_name:
        return None

    mode = _TRANSLATION_POLICY["mode"]
    timeout_s = _TRANSLATION_POLICY["timeout_s"]
    if mode == "online_only":
        return None

    timeout = None if mode == "offline" else timeout_s  # None → blocking load
    pipe = _load_marian_with_timeout(model_name, timeout)
    if pipe is None:
        _MT_FAILED.add(lang)  # remember failure to avoid repeated stalls
        return None

    _MT_CACHE[lang] = pipe
    return pipe


# ===== Online translators (patched) =====
def _libre_translate(text: str, src: str) -> Optional[str]:
    """
    Try multiple LibreTranslate mirrors and both JSON and form payloads.
    """
    bases = [
        "https://libretranslate.com/translate",
        "https://libretranslate.de/translate",
        "https://translate.astian.org/translate",  # extra popular mirror
    ]
    for base in bases:
        # 1) JSON payload
        try:
            r = requests.post(
                base,
                json={"q": text, "source": src, "target": "en", "format": "text"},
                timeout=15,
            )
            if r.status_code == 200:
                out = r.json().get("translatedText")
                if out:
                    return out.strip()
        except Exception:
            pass
        # 2) Form payload (some instances prefer this)
        try:
            r = requests.post(
                base,
                data={"q": text, "source": src, "target": "en", "format": "text"},
                timeout=15,
            )
            if r.status_code == 200:
                out = r.json().get("translatedText")
                if out:
                    return out.strip()
        except Exception:
            pass
    return None


def _mymemory_translate(text: str, src: str) -> Optional[str]:
    """
    MyMemory online fallback. Accepts src='auto' as well.
    """
    try:
        params = {"q": text, "langpair": f"{src}|en"}
        r = requests.get("https://api.mymemory.translated.net/get", params=params, timeout=15)
        if r.status_code == 200:
            js = r.json()
            out = js.get("responseData", {}).get("translatedText")
            if out:
                return str(out).strip()
    except Exception:
        pass
    return None


# ---------------- Language detect + translate to English ----------------
def detect_and_translate(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"error": "No text provided."}

    # detect
    src = "auto"
    lang_prob = None
    try:
        langs = detect_langs(text)
        top = max(langs, key=lambda x: x.prob)
        src = top.lang.lower()
        lang_prob = float(top.prob)
    except Exception:
        src = "auto"

    # try offline (policy-aware) → fallback online (LibreTranslate → MyMemory)
    english, source = None, None
    mt = _get_mt_pipeline(src) if src != "auto" else None
    if mt is not None:
        try:
            english = mt(text, max_length=512)[0]["translation_text"].strip()
            source = "marian-mt"
        except Exception:
            english = None

    if english is None:
        english = _libre_translate(text, src if src != "auto" else "auto")
        if english:
            source = "libretranslate"

    if english is None:
        english = _mymemory_translate(text, src if src != "auto" else "auto")
        if english:
            source = "mymemory"

    out = {
        "language": src.upper() if src != "auto" else "UNKNOWN",
        "language_prob": lang_prob,
        "english": english,
        "source": source,
    }
    if english is None:
        out["error"] = "Translation unavailable (offline model timed out/failed and online fallbacks failed)."
    return out


# ---------------- Sentiment (ensemble: ML + VADER + AFINN) ----------------
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Pipeline:
      1) Auto-translate to English when needed (uses the same policy as Detect & Translate).
      2) Light normalization (emoji mapping, de-elongation).
      3) Heads: your ML model, VADER, and AFINN.
      4) Decision:
         - if ML prob >= 0.70 → ML
         - elif VADER conf >= 0.60 → VADER
         - elif AFINN conf >= 0.55 → AFINN
         - else combine scores:
             S = 0.5*sign(ML)*ML_prob + 0.3*VADER_compound + 0.2*AFINN_norm + emotion_boost
           label = sign(S) with ±0.15 margin; confidence = max(|S|, ML_prob, |VADER|, |AFINN|)
    """
    original = (text or "").strip()
    if not original:
        return {"label": "neutral", "prob": 0.0, "source": "none"}

    # 1) language normalize
    try:
        langs = detect_langs(original)
        lang = max(langs, key=lambda x: x.prob).lang.lower()
    except Exception:
        lang = "en"

    normalized_text = original
    if lang != "en":
        tr = detect_and_translate(original)
        if tr.get("english"):
            normalized_text = tr["english"]

    # 2) light normalization
    normalized_text = _normalize_for_sentiment(normalized_text)

    # 3) heads
    ml_label, ml_prob = analyze_sentiment_label_proba(normalized_text)
    vader = _vader_label_prob(normalized_text)
    afinn = _afinn_score(normalized_text)

    # 4) decision rules
    if ml_prob >= 0.70:
        return {"label": ml_label, "prob": float(ml_prob), "source": "ml"}

    if vader and vader[1] >= 0.60:
        return {"label": vader[0], "prob": float(vader[1]), "source": "vader"}

    if afinn and afinn[1] >= 0.55:
        return {"label": afinn[0], "prob": float(afinn[1]), "source": "afinn"}

    # combine heads
    sign_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    ml_sign = sign_map.get(ml_label, 0.0)
    vader_comp = vader[2] if vader else 0.0
    af_norm = afinn[2] if afinn else 0.0
    boost = _emotion_boost(normalized_text)

    S = 0.5 * ml_sign * float(ml_prob) + 0.3 * float(vader_comp) + 0.2 * float(af_norm) + float(boost)

    if S > 0.15:
        lab = "positive"
    elif S < -0.15:
        lab = "negative"
        # else neutral
    else:
        lab = "neutral"

    conf = float(min(1.0, max(abs(S), float(ml_prob), abs(vader_comp), abs(af_norm))))
    return {"label": lab, "prob": conf, "source": "ensemble"}


# ---------------- Keyphrases (TF-IDF) ----------------
def extract_keyphrases(text: str, top_k: int = 10, max_ngram: int = 3) -> List[str]:
    if not text or not text.strip():
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, max(1, int(max_ngram))))
    X = vec.fit_transform([text])
    scores = X.toarray()[0]
    feats = np.array(vec.get_feature_names_out())
    order = np.argsort(-scores)
    return [feats[i] for i in order[: max(1, int(top_k))]]


# ---------------- TextRank Summarizer ----------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(t: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(t or "") if s.strip()]


def textrank_summarize(text: str, n_sent: int = 5) -> List[str]:
    sents = _split_sentences(text or "")
    if not sents:
        return []
    if len(sents) <= n_sent:
        return sents

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(sents)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)

    G = nx.from_numpy_array(sim)
    scores = nx.pagerank(G)
    ranked = sorted(((scores[i], i) for i in range(len(sents))), reverse=True)
    keep_idx = sorted([i for _, i in ranked[: max(1, int(n_sent))]])
    return [sents[i] for i in keep_idx]


# ---------------- RAG-lite (BM25 with TF-IDF fallback) ----------------
_BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    _BM25_AVAILABLE = False

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def _tokenize(t: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(_normalize_text(t))]


def _chunk_text(t: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    t = _normalize_text(t)
    if not t:
        return []
    chunks, start = [], 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        chunks.append(t[start:end])
        if end == len(t):
            break
        start = max(0, end - overlap)
    return chunks


def build_rag_index_from_texts(texts: List[str], chunk_size: int = 800, overlap: int = 120) -> Dict[str, Any]:
    chunks: List[str] = []
    for t in texts:
        chunks.extend(_chunk_text(t, chunk_size=chunk_size, overlap=overlap))
    index: Dict[str, Any] = {"chunks": chunks}
    if not chunks:
        index.update({"bm25": None, "tokens": None, "tfidf_vec": None, "tfidf_mat": None})
        return index

    if _BM25_AVAILABLE:
        tokenized = [_tokenize(c) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        index.update({"bm25": bm25, "tokens": tokenized})
    else:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(chunks)
        index.update({"bm25": None, "tokens": None, "tfidf_vec": vec, "tfidf_mat": X})
    return index


def _top_k_idxs_bm25(query: str, index: Dict[str, Any], k: int):
    q_toks = _tokenize(query)
    scores = index["bm25"].get_scores(q_toks)
    order = np.argsort(-scores)
    return order[:k], scores


def _top_k_idxs_tfidf(query: str, index: Dict[str, Any], k: int):
    qv = index["tfidf_vec"].transform([query])
    sims = cosine_similarity(qv, index["tfidf_mat"])[0]
    order = np.argsort(-sims)
    return order[:k], sims


def _mmr_select_sentences(query: str, candidates: List[str], top_n: int = 4, diversity: float = 0.35) -> List[str]:
    """Simple MMR over TF-IDF vectors to avoid redundancy in extractive answers."""
    if not candidates:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(candidates)
    q = vec.transform([query])
    sim_to_query = cosine_similarity(q, X)[0]
    sim_between = cosine_similarity(X)

    selected, remaining = [], list(range(len(candidates)))
    while remaining and len(selected) < top_n:
        if not selected:
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            remaining.remove(idx)
            continue
        best, best_score = None, -1e9
        for i in remaining:
            redundancy = max(sim_between[i, j] for j in selected) if selected else 0.0
            score = (1 - diversity) * sim_to_query[i] - diversity * redundancy
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


def answer_from_docs(query: str, index: Optional[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    if not index or not index.get("chunks"):
        return {"error": "No documents indexed. Upload one or more .txt/.pdf files first."}

    chunks = index["chunks"]
    use_bm25 = index.get("bm25") is not None

    if use_bm25:
        order, scores = _top_k_idxs_bm25(query, index, max(1, int(top_k)))
    else:
        order, scores = _top_k_idxs_tfidf(query, index, max(1, int(top_k)))

    matches = []
    gathered_sents: List[str] = []
    for rank, i in enumerate(order, start=1):
        score = float(scores[i]) if len(scores) > i else 0.0
        snippet = (chunks[i][:500]).replace("\n", " ")
        matches.append({"rank": rank, "score": score, "snippet": snippet})
        gathered_sents.extend(_split_sentences(chunks[i])[:12])

    answer_sents = _mmr_select_sentences(query, gathered_sents, top_n=4, diversity=0.35)
    return {
        "matches": matches,
        "extractive_answer": " ".join(answer_sents) if answer_sents else "",
    }
