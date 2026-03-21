"""
chain.py — Adaptive RAG chain with query-type-specific prompts
==============================================================

Three prompts, one per query level:

  BROAD   → instructs the LLM to produce a structured overview / summary
  MEDIUM  → instructs the LLM to focus on a specific topic using section summaries
  NARROW  → instructs the LLM to give a precise, direct answer from verbatim chunks

The public entry point is adaptive_ask(), which:
  1. Classifies the question (broad / medium / narrow)
  2. Retrieves context from the correct Pinecone namespace
  3. Feeds it into the matching prompt
  4. Returns the answer + metadata about which level was used
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL, GROQ_API_KEY
from query_classifier import classify_query
from retriever import get_adaptive_retriever

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM
# ─────────────────────────────────────────────────────────────────────────────

def get_llm() -> ChatGroq:
    print(f"  [LLM] Loading Groq model: {LLM_MODEL}")
    llm = ChatGroq(model=LLM_MODEL)
    print(f"  [LLM] ✅ LLM ready")
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# 2. THREE PROMPTS — one per query level
# ─────────────────────────────────────────────────────────────────────────────

_BROAD_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant answering questions about a YouTube video.
The context below is a comprehensive summary of the ENTIRE video.

Your task: give a well-structured, thorough overview that directly answers the question.

Guidelines:
- Cover the main topic, key arguments, important people/entities, and the conclusion.
- Use short paragraphs or bullet points where they aid clarity.
- Do NOT mention summaries, transcripts, or retrieval systems.
- If the context does not contain enough information, say:
  "I don't know based on the available information."

Context (full-video summary):
{context}

Question: {question}

Answer:
""",
)

_MEDIUM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant answering questions about a YouTube video.
The context below contains summaries of the most relevant sections of the video.

Your task: answer the question using ONLY the provided section summaries.

Guidelines:
- Focus specifically on the topic, concept, or person the question asks about.
- Combine information from multiple sections when relevant.
- Be factual and specific — preserve names, numbers, and claims from the context.
- Do NOT mention summaries, transcripts, or retrieval systems.
- If the answer is not present, say:
  "I don't know based on the available information."

Context (relevant section summaries):
{context}

Question: {question}

Answer:
""",
)

_NARROW_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant answering precise questions about a YouTube video.
The context below contains verbatim excerpts from the exact part of the video
most relevant to the question.

Your task: give a direct, concise, and precise answer.

Guidelines:
- Answer only what is clearly stated in the context.
- If it is a timestamp question, describe what was said or happened at that moment.
- Do NOT paraphrase loosely — stay close to the actual content.
- Do NOT mention transcripts, captions, or retrieval systems.
- If the answer is not present, say:
  "I don't know based on the available information."

Context (verbatim transcript excerpts):
{context}

Question: {question}

Answer:
""",
)

_PROMPTS = {
    "broad":  _BROAD_PROMPT,
    "medium": _MEDIUM_PROMPT,
    "narrow": _NARROW_PROMPT,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _format_docs(docs: list) -> str:
    """Merge retrieved Document objects into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain(retriever, query_type: str = "medium"):
    """
    Build a RAG chain for a specific query type.

    Flow: question → parallel(retrieve|passthrough) → prompt → LLM → parse

    Args:
        retriever  : A LangChain retriever (already scoped to the right namespace).
        query_type : "broad", "medium", or "narrow" — selects the matching prompt.

    Returns:
        A runnable LangChain chain.
    """
    print(f"  [Chain] Building {query_type.upper()} RAG chain...")

    prompt = _PROMPTS.get(query_type, _MEDIUM_PROMPT)
    llm    = get_llm()
    parser = StrOutputParser()

    parallel = RunnableParallel({
        "context":  retriever | RunnableLambda(_format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | parser
    print(f"  [Chain] ✅ {query_type.upper()} chain ready")
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# 4. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_ask(embedding, video_id: str, question: str) -> dict:
    """
    Full adaptive RAG pipeline:
      1. Classify the question → broad / medium / narrow
      2. Retrieve context from the matching Pinecone namespace
      3. Run the matching prompt through the LLM
      4. Return the answer + metadata

    Args:
        embedding : The embedding model singleton.
        video_id  : YouTube video ID (Pinecone namespace root).
        question  : The user's question string.

    Returns:
        {
            "answer":     str,    # the LLM's answer
            "query_type": str,    # "broad" | "medium" | "narrow"
            "video_id":   str,
            "question":   str,
        }
    """
    print(f"\n  [Chain] ❓ Question: '{question}'")

    # Step 1 — classify
    query_type = classify_query(question)

    # Step 2 — retrieve from the right namespace
    retriever  = get_adaptive_retriever(embedding, video_id=video_id, query_type=query_type)

    # Step 3 — build the matching chain and invoke
    chain  = build_chain(retriever, query_type=query_type)
    answer = chain.invoke(question)

    print(f"  [Chain] ✅ Answer ({query_type}, {len(answer)} chars)")
    return {
        "answer":     answer,
        "query_type": query_type,
        "video_id":   video_id,
        "question":   question,
    }


# ── Legacy shim — keeps old callers working ───────────────────────────────────

def ask(chain, question: str) -> str:
    """Legacy helper. New code should use adaptive_ask() instead."""
    print(f"  [Chain] ⚠️  ask() is deprecated. Use adaptive_ask().")
    answer = chain.invoke(question)
    return answer
