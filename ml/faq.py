# ml/faq.py
from __future__ import annotations
import os, json
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = "data/faq.jsonl"

def _default_faq() -> List[Dict]:
    return [
        {"q":"how do i set the gemini api key?","a":"Paste it in the app sidebar or set GEMINI_API_KEY environment variable."},
        {"q":"what tools are available?","a":"Weather checker, Currency converter (USD↔NPR), Sentiment analyzer, Language detector, and App FAQ search."},
        {"q":"does usd to npr work?","a":"Yes. We normalize 'nrs'→'NPR' and use multi-API fallbacks."},
        {"q":"can i retrain models?","a":"Yes, retrain intent/sentiment from the sidebar if you provide labeled data."},
    ]

def load_faq() -> List[Dict]:
    if not os.path.exists(FAQ_PATH):
        os.makedirs(os.path.dirname(FAQ_PATH), exist_ok=True)
        with open(FAQ_PATH, "w", encoding="utf-8") as f:
            for row in _default_faq():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out = []
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def retrieve_faq(query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
    faq = load_faq()
    corpus = [f"{x['q']} || {x['a']}" for x in faq]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    ranked = sorted([(float(sims[i]), faq[i]) for i in range(len(faq))], key=lambda t: t[0], reverse=True)
    return ranked[:max(1, top_k)]
