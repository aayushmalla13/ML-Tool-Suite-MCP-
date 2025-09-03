# ml/slots.py
import re
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None  # optional; app will work without spaCy

ISO = {"USD","NPR","EUR","GBP","INR","JPY","AUD","CAD","CNY","CHF","SEK","BRL","PHP","MXN"}
SYN = {"dollar":"USD","us dollar":"USD","bucks":"USD",
       "rupee":"INR","nepali rupee":"NPR","nepalese rupee":"NPR","rupees":"INR",
       "euro":"EUR","pound":"GBP","yen":"JPY","yuan":"CNY","real":"BRL","peso":"MXN"}  # generic 'peso' mapped, still ambiguous IRL

CURR_RE = re.compile(r"\b([A-Z]{3})\b", re.I)

def _normalize_currency(tok: str):
    if not tok: return None
    t = tok.strip().lower()
    if t.upper() in ISO: return t.upper()
    return SYN.get(t)

def extract_city(text: str):
    # Prefer NER if available
    if _NLP is not None:
        doc = _NLP(text)
        for ent in doc.ents:
            if ent.label_ in ("GPE","LOC"):
                return ent.text
    # fallback heuristic: "in <Word>"
    m = re.search(r"\bin\s+([A-Z][a-zA-Z]+)", text)
    return m.group(1) if m else None

def extract_currencies(text: str):
    out = []
    # ISO codes
    for m in CURR_RE.finditer(text):
        c = _normalize_currency(m.group(1))
        if c and c not in out: out.append(c)
    # synonyms
    low = text.lower()
    for k,v in SYN.items():
        if re.search(rf"\b{re.escape(k)}\b", low) and v not in out:
            out.append(v)
    # common typo
    if "nrs" in low and "NPR" not in out:
        out.append("NPR")
    return out[:2]

def extract_slots(text: str):
    city = extract_city(text)
    currs = extract_currencies(text)
    from_c = currs[0] if len(currs) > 0 else None
    to_c   = currs[1] if len(currs) > 1 else None
    return {"city": city, "from_currency": from_c, "to_currency": to_c}
