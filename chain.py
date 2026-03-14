import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, GROQ_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def get_llm() -> ChatGroq:
    """Initialize and return the Groq LLM."""
    print(f"  [LLM] Loading Groq model: {LLM_MODEL}")
    llm = ChatGroq(model=LLM_MODEL)
    print(f"  [LLM] ✅ LLM ready")
    return llm


def get_prompt() -> PromptTemplate:
    """Return the RAG prompt template."""
    return PromptTemplate(
        template="""
You are a helpful assistant that answers questions about YouTube videos.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}

Answer:
        """,
        input_variables=["context", "question"]
    )


def format_docs(retrieved_docs: list) -> str:
    """Merge retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_chain(retriever):
    """
    Build and return the full RAG chain.
    Flow: question → (retrieve + passthrough) → prompt → llm → parse
    """
    print("  [Chain] Building RAG chain...")

    llm    = get_llm()
    prompt = get_prompt()
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context" : retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    chain = parallel_chain | prompt | llm | parser
    print("  [Chain] ✅ RAG chain built successfully")
    return chain


def ask(chain, question: str) -> str:
    """
    Run the chain with a question and return the answer string.
    """
    print(f"\n  [Chain] 🔍 Question: '{question}'")
    print(f"  [Chain] Invoking chain...")
    answer = chain.invoke(question)
    print(f"  [Chain] ✅ Answer received ({len(answer)} characters)")
    return answer
