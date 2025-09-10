# mcp_server.py
# Minimal MCP server exposing your ML tools (summarize, keyphrases, sentiment, detect_translate).
# Compatible with either:
#   - Official SDK:  pip install -U "mcp[cli]"
#   - Community pkg: pip install -U fastmcp

import sys

# ---- try both FastMCP import paths ----
FastMCP = None
err_msgs = []
try:
    from mcp.server.fastmcp import FastMCP  # official SDK
except Exception as e:
    err_msgs.append(f"official mcp[cli] import failed: {e!r}")
    try:
        from fastmcp import FastMCP  # community package
    except Exception as e2:
        err_msgs.append(f"community fastmcp import failed: {e2!r}")

if FastMCP is None:
    sys.stderr.write(
        "\n[ERROR] Could not import FastMCP from either package.\n"
        "Install one of these in the SAME Python env you’ll run with MCP Inspector:\n"
        "  - pip install -U \"mcp[cli]\"   (recommended)\n"
        "  - pip install -U fastmcp       (community)\n"
        f"Details: {err_msgs}\n\n"
    )
    sys.exit(1)

# ---- your tool functions from tools.py ----
from tools import (
    textrank_summarize,
    extract_keyphrases,
    analyze_sentiment,
    detect_and_translate,
)

# ---- create server ----
mcp = FastMCP("ml_tool_suite")

@mcp.tool()
def summarize(text: str, n_sentences: int = 5) -> dict:
    """Extractive summary using TextRank. Returns {'sentences': [...]}."""
    return {"sentences": textrank_summarize(text or "", n_sent=int(n_sentences))}

@mcp.tool()
def keyphrases(text: str, top_k: int = 10, max_ngram: int = 3) -> dict:
    """Top TF-IDF phrases. Returns {'phrases': [...]}."""
    return {"phrases": extract_keyphrases(text or "", top_k=int(top_k), max_ngram=int(max_ngram))}

@mcp.tool()
def sentiment(text: str) -> dict:
    """Coarse sentiment (positive/negative/neutral) with confidence."""
    return analyze_sentiment(text or "")

@mcp.tool()
def detect_translate(text: str) -> dict:
    """Detect language and translate to English (offline/online based on tools.py policy)."""
    return detect_and_translate(text or "")

if __name__ == "__main__":
    # STDIO server (Inspector will spawn this process)
    mcp.run()
