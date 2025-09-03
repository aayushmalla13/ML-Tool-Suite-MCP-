# build_prompt.py
from typing import List, Optional

def build_mcp_prompt(system: str, tools: str, query: str, memory: Optional[List[str]] = None) -> str:
    memory = memory or []

    parts = []
    parts.append(f"System:\n{system}\n")
    parts.append(f"Available Tools:\n{tools}\n")
    if memory:
        parts.append("Memory:\n" + "\n".join(memory) + "\n")
    parts.append(f"User Query:\n{query}\n")

    parts.append(
        "Instructions:\n"
        "- If the user asks questions about their uploaded document OR asks something that requires information from their docs, respond ONLY with a code block:\n"
        "```tool_code\nanswer_from_docs(query='QUESTION', top_k=5)\n```\n"
        "- If the user asks to summarize text, respond ONLY with a code block:\n"
        "```tool_code\ntextrank_summarize(text='TEXT', n_sent=5)\n```\n"
        "- If the user wants keywords/keyphrases, respond ONLY with a code block:\n"
        "```tool_code\nextract_keyphrases(text='TEXT', top_k=8, max_ngram=3)\n```\n"
        "- If the user wants the language OR asks to translate non-English text to English (e.g., 'what language is this', 'translate to English', 'meaning in English'), respond ONLY with a code block:\n"
        "```tool_code\ndetect_and_translate(text='TEXT')\n```\n"
        "- If the user wants sentiment, respond ONLY with a code block:\n"
        "```tool_code\nanalyze_sentiment(text='TEXT')\n```\n"
        "- Otherwise, reply normally.\n"
        "Be strict: when one of the above matches, return ONLY the `tool_code` block."
    )
    return "\n".join(parts)
