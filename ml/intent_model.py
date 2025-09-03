# ml/intent_model.py
from __future__ import annotations
import os, json
from typing import List, Tuple, Iterable, Optional
from collections import Counter

from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ARTIFACT_PATH = "artifacts/intent_model.joblib"
USER_DATA = "data/intent_labeled.jsonl"
SEED = 7

def _seed_data() -> List[Tuple[str, str]]:
    return [
        ("what's the weather in kathmandu", "weather"),
        ("weather forecast for pokhara this week", "weather"),
        ("how cold is it in new york today", "weather"),
        ("usd to npr rate", "fx"),
        ("convert 100 usd to eur", "fx"),
        ("what is the exchange rate from gbp to inr", "fx"),
        ("tell me a joke", "other"),
        ("who are you", "other"),
        ("explain large language models", "other"),
    ]

def _load_user_data(path: str) -> Iterable[Tuple[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            txt = obj.get("text", "").strip()
            lab = obj.get("label", "").strip()
            if txt and lab:
                yield (txt, lab)

def _choose_cv(y: List[str]) -> Optional[int]:
    """Return 3, 2, or None depending on minimum class count."""
    counts = Counter(y)
    min_count = min(counts.values())
    if min_count >= 3:
        return 3
    if min_count >= 2:
        return 2
    return None  # too few for calibration

def train_intent_model(save_path: str = ARTIFACT_PATH) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    seed = _seed_data()
    seed.extend(list(_load_user_data(USER_DATA)))

    X = [t for t, _ in seed]
    y = [l for _, l in seed]

    if len(set(y)) < 2:
        # not enough classes → still fit a model but warn
        print("Warning: Only one class present; training a degenerate classifier.")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y if len(set(y)) > 1 else None, random_state=SEED)

    base = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    cv = _choose_cv(y_tr)
    if cv is None:
        # Too few samples per class for calibration → plain LR
        base.fit(X_tr, y_tr)
        clf = base
        print("[intent] Calibration skipped (min class count < 2).")
    else:
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
        clf.fit(X_tr, y_tr)
        print(f"[intent] Calibrated with cv={cv}.")

    if len(set(y_te)) > 1:
        print(classification_report(y_te, clf.predict(X_te)))
    else:
        print("Note: Test set had <2 classes; skipping classification report.")

    dump(clf, save_path)
    return save_path

# ---------- Inference ----------
_model = None

def load_intent_model(path: str = ARTIFACT_PATH):
    global _model
    if _model is None and os.path.exists(path):
        _model = load(path)
    return _model

def predict_intent(text: str, threshold: float = 0.65) -> Tuple[str, float]:
    """
    Returns (label, prob). If model missing or top prob < threshold -> ('other', prob).
    Works for both calibrated and plain LR pipelines.
    """
    m = load_intent_model()
    if m is None:
        return "other", 0.0
    probs = m.predict_proba([text])[0]
    labels = m.classes_
    top_idx = probs.argmax()
    label = labels[top_idx]
    prob = float(probs[top_idx])
    if prob < threshold:
        return "other", prob
    return label, prob
