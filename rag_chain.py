"""
rag_chain.py — RAG chain using Groq (Llama 3.1) + pre-built FAISS index.

HOW API KEY INJECTION WORKS:
  os.environ.get("GROQ_API_KEY") reads the key from the process environment.
  On HuggingFace Spaces, you set this in:
    Space Settings → Variables and Secrets → New Secret → GROQ_API_KEY
  HF injects it as an environment variable at container startup.
  The key NEVER appears in your code or git history — only in HF's encrypted vault.
  This is identical to how AWS Secrets Manager, GCP Secret Manager, and
  Azure Key Vault work: code references a name, runtime provides the value.
"""

import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K_CHUNKS = 4

BGP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Cisco network engineer specialising in BGP troubleshooting.\n"
     "Answer questions based only on the provided documentation and telemetry logs.\n"
     "Always suggest the relevant show command to verify the issue.\n"
     "If the answer is not in the context, say clearly: "
     "'This information is not in the knowledge base.'\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
])


# ── Load embeddings (cached after first call) ─────────────────────────────────

_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return cached HuggingFaceEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Load FAISS vector store ───────────────────────────────────────────────────

_vectorstore = None

def _build_index():
    """Build FAISS index from data/ files. Called automatically if index is missing."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("Building FAISS index from documents...")
    all_docs = []
    for data_dir in ["data/docs", "data/telemetry"]:
        if not os.path.exists(data_dir):
            continue
        loader = DirectoryLoader(
            data_dir, glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}, show_progress=False,
        )
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    vs = FAISS.from_documents(chunks, get_embeddings())
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"Index built — {len(chunks)} chunks saved to {INDEX_DIR}/")
    return vs


def get_vectorstore() -> FAISS:
    """Load FAISS index, building it first if it doesn't exist."""
    global _vectorstore
    if _vectorstore is None:
        if not Path(INDEX_DIR).exists():
            # Auto-build on first run (e.g. HF Spaces where binary files can't be committed)
            _vectorstore = _build_index()
        else:
            _vectorstore = FAISS.load_local(
                INDEX_DIR,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
    return _vectorstore


# ── Build RAG chain ───────────────────────────────────────────────────────────

_qa_chain = None

def _format_docs(docs: list) -> str:
    """Concatenate document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_chain():
    """Build and cache the RAG chain using pure LCEL (langchain-core only)."""
    global _qa_chain
    if _qa_chain is None:
        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=1024,
        )

        retriever = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_CHUNKS},
        )

        # LCEL chain — runs retriever and passthrough in parallel, then calls LLM
        # RunnableParallel fetches context docs AND passes the question through simultaneously
        # Result: {"answer": "...", "context": [doc, doc, ...]}
        _qa_chain = RunnableParallel(
            answer=(
                {"context": retriever | _format_docs, "input": RunnablePassthrough()}
                | BGP_PROMPT
                | llm
                | StrOutputParser()
            ),
            context=retriever,
        )
    return _qa_chain


# ── Public API ────────────────────────────────────────────────────────────────

def query_bgp(question: str) -> dict:
    """
    Run a BGP question through the RAG pipeline.

    Returns:
        {
            "answer": str,
            "sources": list[str],   # Unique filenames of retrieved docs
            "source_documents": list # Raw LangChain Document objects
        }
    """
    chain = get_qa_chain()
    # LCEL chain takes the question string directly
    # Returns {"answer": str, "context": list[Document]}
    result = chain.invoke(question)

    source_docs = result.get("context", [])
    sources = list(dict.fromkeys(
        Path(doc.metadata.get("source", "unknown")).name
        for doc in source_docs
    ))

    return {
        "answer": result["answer"],
        "sources": sources,
        "source_documents": source_docs,
    }


def get_chunk_count() -> int:
    """Return total number of chunks in the FAISS index (for sidebar display)."""
    try:
        vs = get_vectorstore()
        return vs.index.ntotal
    except Exception:
        return 0


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set.")
        print("   Export it: export GROQ_API_KEY=gsk_...")
        sys.exit(1)

    print("Loading RAG chain...")
    test_question = "Why is my BGP neighbor stuck in IDLE state?"
    print(f"\nQuestion: {test_question}\n")

    result = query_bgp(test_question)
    print("Answer:")
    print(result["answer"])
    print(f"\nSources retrieved: {result['sources']}")
