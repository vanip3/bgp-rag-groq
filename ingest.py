"""
ingest.py — Build FAISS vector index from BGP documentation.

Run this ONCE locally before deploying to HuggingFace Spaces:
    python ingest.py

Output: faiss_index/index.faiss + faiss_index/index.pkl
Commit both files to your repo — the Space loads them at startup.

WHY PRE-BUILD?
- Building the index requires downloading ~90MB sentence-transformers model
  and embedding every chunk. On HF free CPU tier this takes 3-5 minutes
  and risks OOM errors. Pre-building means the Space starts in ~3 seconds.
- Pattern: compute expensive artifacts locally, ship the result.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIRS = ["data/docs", "data/telemetry"]
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ── Load documents ────────────────────────────────────────────────────────────

def load_documents(data_dirs: list[str]) -> list:
    """Load all .txt files from the given directories."""
    all_docs = []
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"  ⚠️  Directory not found, skipping: {data_dir}")
            continue
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        docs = loader.load()
        print(f"  📂 {data_dir}: {len(docs)} file(s) loaded")
        all_docs.extend(docs)
    return all_docs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BGP RAG — FAISS Index Builder")
    print("=" * 60)

    # 1. Load raw documents
    print("\n[1/4] Loading documents...")
    docs = load_documents(DATA_DIRS)
    if not docs:
        raise RuntimeError("No documents found. Check data/ directory.")
    print(f"  ✅ Total documents loaded: {len(docs)}")

    # 2. Split into chunks
    print(f"\n[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  ✅ Chunks created: {len(chunks)}")
    print(f"  📊 Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    # 3. Create embeddings
    print(f"\n[3/4] Loading embedding model: {EMBEDDING_MODEL}")
    print("  (First run downloads ~90MB model — subsequent runs use cache)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  ✅ Embedding model loaded")

    # 4. Build and save FAISS index
    print(f"\n[4/4] Building FAISS index and saving to {INDEX_DIR}/...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    # Verify files were written
    index_file = Path(INDEX_DIR) / "index.faiss"
    pkl_file = Path(INDEX_DIR) / "index.pkl"
    if index_file.exists() and pkl_file.exists():
        size_mb = (index_file.stat().st_size + pkl_file.stat().st_size) / (1024 * 1024)
        print(f"  ✅ Saved: {INDEX_DIR}/index.faiss")
        print(f"  ✅ Saved: {INDEX_DIR}/index.pkl")
        print(f"  📦 Total index size: {size_mb:.2f} MB")
    else:
        raise RuntimeError("Index files not found after save — check permissions.")

    print("\n" + "=" * 60)
    print("✅ FAISS index built successfully!")
    print(f"   Documents: {len(docs)}")
    print(f"   Chunks:    {len(chunks)}")
    print(f"   Location:  {INDEX_DIR}/")
    print("\nNext steps:")
    print("  1. git add faiss_index/")
    print("  2. git commit -m 'Add pre-built FAISS index'")
    print("  3. Push to HuggingFace Space")
    print("=" * 60)


if __name__ == "__main__":
    main()
