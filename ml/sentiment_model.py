# ml/sentiment_model.py
from __future__ import annotations
import os, json, concurrent.futures
from typing import Iterable, Tuple, List, Optional

import numpy as np
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional tiny transformer head (POS/NEG). Neutral handled by classical model.
_TRANSFORMERS_OK = True
try:
    from transformers import pipeline  # type: ignore
except Exception:
    _TRANSFORMERS_OK = False

ARTIFACT_PATH = "artifacts/sentiment_lr_calibrated.joblib"
USER_DATA = "data/sentiment_labeled.jsonl"
SEED = 13
CLASSES = ["negative", "neutral", "positive"]

# ---------- Seed data (very small; grows with your labeled data) ----------
def _seed_data() -> List[Tuple[str, str]]:
    return [
        # positive
        ("i love this", "positive"),
        ("this is fantastic and great", "positive"),
        ("absolutely amazing job", "positive"),
        ("i'm happy with the result", "positive"),
        # negative
        ("i hate this", "negative"),
        ("this is terrible and awful", "negative"),
        ("i'm angry about the service", "negative"),
        ("so disappointed and frustrated", "negative"),
        # neutral
        ("this is a pen", "neutral"),
        ("the meeting is at 2pm", "neutral"),
        ("it might rain tomorrow", "neutral"),
        ("please see the attached file", "neutral"),
    ]

def _load_user_data(path: str) -> Iterable[Tuple[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = (obj.get("text") or "").strip()
            lab = (obj.get("label") or "").strip().lower()
            if txt and lab in CLASSES:
                yield (txt, lab)

# ---------- Train / load classical model ----------
def train_sentiment_model(save_path: str = ARTIFACT_PATH) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = _seed_data()
    data.extend(list(_load_user_data(USER_DATA)))

    X = [t for t, _ in data]
    y = [l for _, l in data]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    base = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    # Calibrate probs for better thresholds
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_tr, y_tr)

    # For your console / streamlit logs
    try:
        print(classification_report(y_te, clf.predict(X_te), digits=3))
    except Exception:
        pass

    dump(clf, save_path)
    return save_path

_model: Pipeline | None = None
def _load_lr_model(path: str = ARTIFACT_PATH) -> Pipeline:
    global _model
    if _model is None:
        if not os.path.exists(path):
            train_sentiment_model(path)
        _model = load(path)
    return _model

# ---------- Optional transformer head (POS/NEG) with timeout ----------
_HF_PIPE = None
_HF_DISABLED = False

def _load_hf_with_timeout(timeout_s: int = 10):
    if not _TRANSFORMERS_OK:
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(pipeline, "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        try:
            return fut.result(timeout=timeout_s)
        except Exception:
            try:
                fut.cancel()
            except Exception:
                pass
            return None

def _get_hf_pipe() -> Optional[any]:
    global _HF_PIPE, _HF_DISABLED
    if _HF_DISABLED:
        return None
    if _HF_PIPE is not None:
        return _HF_PIPE
    pipe = _load_hf_with_timeout(timeout_s=10)
    if pipe is None:
        _HF_DISABLED = True
        return None
    _HF_PIPE = pipe
    return _HF_PIPE

# ---------- Public inference API ----------
def analyze_sentiment_label_proba(text: str) -> Tuple[str, float]:
    """
    Returns (label, prob) where label ∈ {'negative','neutral','positive'} and prob ∈ [0,1].
    Strategy:
      - Try tiny transformer (pos/neg). If high confidence (>0.85), return it.
      - Else use calibrated LR (3-class) and return its label+prob.
    """
    t = (text or "").strip()
    if not t:
        return "neutral", 0.0

    # 1) optional transformer head
    pipe = _get_hf_pipe()
    if pipe is not None and len(t.split()) <= 200:
        try:
            out = pipe(t)[0]  # {'label':'POSITIVE'|'NEGATIVE', 'score':float}
            lbl = (out.get("label") or "").upper()
            score = float(out.get("score") or 0.0)
            if lbl in {"POSITIVE", "NEGATIVE"} and score >= 0.85:
                return ("positive" if lbl == "POSITIVE" else "negative", score)
        except Exception:
            pass  # fall through to LR

    # 2) calibrated LR (handles neutral)
    lr = _load_lr_model()
    proba = lr.predict_proba([t])[0]
    idx = int(np.argmax(proba))
    label = lr.classes_[idx]
    return label, float(proba[idx])
