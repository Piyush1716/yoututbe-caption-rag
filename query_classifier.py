"""
query_classifier.py — Classify a user question into one of three retrieval levels
==================================================================================

Query types
-----------
  broad   "What is this video about?"           → needs the full-video summary
  medium  "What did Derek say about entropy?"   → needs section-level summaries
  narrow  "What happened at 5:30?"              → needs exact raw transcript chunks

The classifier uses a fast Groq LLM call so it adds <1 s latency while
dramatically improving retrieval quality for every question type.
"""

import os
import re
from langchain_groq import ChatGroq
from config import LLM_MODEL, GROQ_API_KEY

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── Classifier prompt ─────────────────────────────────────────────────────────
_CLASSIFIER_PROMPT = """You are a query classifier for a video Q&A system.

Classify the user's question into exactly ONE of these three categories:

  broad  — The question asks for an overall summary, main theme, general overview,
            or broad understanding of the entire video.
            Examples: "What is this video about?", "Summarise the video",
                      "What are the main topics covered?", "Give me an overview"

  medium — The question asks about a specific topic, concept, person, argument,
            or section — but not an exact moment or timestamp.
            Examples: "What did the speaker say about climate change?",
                      "Explain the part about machine learning",
                      "What arguments were made for X?"

  narrow — The question references a specific timestamp, a very precise detail,
            an exact quote, or asks "what happened at [time]".
            Examples: "What was said at 5:30?", "What exact words did she use?",
                      "What happened right after the intro?"

User question: "{question}"

Reply with ONLY one word — either:  broad  |  medium  |  narrow
No explanation. No punctuation. Just the single word."""


# ── Public API ────────────────────────────────────────────────────────────────

QUERY_TYPES = ("broad", "medium", "narrow")


def classify_query(question: str) -> str:
    """
    Classify a user question and return one of: "broad", "medium", "narrow".

    Falls back to "medium" if the LLM returns something unexpected.

    Args:
        question : The raw user question string.

    Returns:
        One of "broad", "medium", "narrow".
    """
    print(f"  [Classifier] Classifying question: '{question[:80]}...' " if len(question) > 80
          else f"  [Classifier] Classifying question: '{question}'")

    llm = ChatGroq(model=LLM_MODEL, temperature=0)

    prompt  = _CLASSIFIER_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    raw     = response.content.strip().lower()

    # Extract the first recognised keyword from the response
    # (guards against the model adding punctuation or extra words)
    match = re.search(r"\b(broad|medium|narrow)\b", raw)
    if match:
        query_type = match.group(1)
    else:
        print(f"  [Classifier] ⚠️  Unexpected response '{raw}' — defaulting to 'medium'")
        query_type = "medium"

    _ICONS = {"broad": "🌐", "medium": "🔍", "narrow": "🎯"}
    print(f"  [Classifier] ✅ Type: {_ICONS[query_type]} {query_type.upper()}")
    return query_type
