"""
app.py — BGP Troubleshooting RAG Chatbot (Cloud Edition)
Deployed on HuggingFace Spaces. Run: streamlit run app.py

HuggingFace Spaces auto-detects app.py and runs it as the entry point.
The GROQ_API_KEY must be set in Space Settings → Variables and Secrets.

WHY ERROR HANDLING FOR MISSING API KEY MATTERS:
  In cloud deployments, environment variables are set by operators, not users.
  If someone forks your Space and forgets to add their GROQ_API_KEY, the app
  will crash with an ugly traceback. Explicit error handling with st.error()
  gives them a clear, actionable message instead of a 500 error. This is the
  difference between a hobby script and a production app.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads .env locally; no-op on HF Spaces (key comes from Secrets instead)

import streamlit as st
from rag_chain import query_bgp, get_chunk_count, get_vectorstore, get_embeddings

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BGP Troubleshooting RAG — Cloud Edition",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Key Guard (must be first — stops execution if key missing) ─────────────

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "⚠️ **GROQ_API_KEY not set.**\n\n"
        "To deploy your own copy:\n"
        "1. Go to your HuggingFace Space\n"
        "2. Click **Settings** → **Variables and Secrets**\n"
        "3. Add a new Secret: `GROQ_API_KEY` = your key from [console.groq.com](https://console.groq.com)\n\n"
        "For local development: `export GROQ_API_KEY=gsk_...`"
    )
    st.stop()  # Halts rendering — nothing below this runs without the key

# ── Session State Initialization ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_loaded" not in st.session_state:
    st.session_state.chain_loaded = False

# ── Example Questions ─────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "Why is my BGP neighbor stuck in IDLE state?",
    "What does %BGP-5-ADJCHANGE neighbor Down mean?",
    "How do I verify BGP authentication is working?",
    "My BGP session keeps flapping — what should I check?",
    "Based on the telemetry logs, what is the most common failure?",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌐 BGP RAG Chatbot")
    st.caption("Cloud Edition")

    # Deployment info
    st.markdown("### 🚀 Deployment")
    st.markdown(
        """
| Component | Value |
|-----------|-------|
| **LLM** | Groq API — llama-3.1-8b-instant |
| **Embeddings** | all-MiniLM-L6-v2 (Local) |
| **Vector DB** | FAISS (Pre-built index) |
| **Hosted on** | HuggingFace Spaces |
        """
    )

    # Knowledge base stats
    st.markdown("### 📚 Knowledge Base")
    chunk_count = get_chunk_count()
    if chunk_count > 0:
        st.success(f"✅ Index loaded — **{chunk_count:,}** chunks")
    else:
        st.warning("⚠️ Index not loaded yet")

    st.markdown(
        """
**Documents indexed:**
- bgp_states_guide.txt
- bgp_troubleshooting_guide.txt
- bgp_show_commands.txt
- bgp_error_messages.txt
- bgp_configuration_guide.txt
- bgp_telemetry_logs.txt
        """
    )

    # Example questions
    st.markdown("### 💡 Example Questions")
    st.caption("Click to populate the input")

    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
            st.session_state["pending_question"] = q

    # Footer
    st.divider()
    st.caption(
        "🔗 [Local Version (nettune)](https://github.com/vanip3/bgp-rag-chatbot)  \n"
        "Powered by [Groq](https://groq.com) · [LangChain](https://langchain.com) · [FAISS](https://github.com/facebookresearch/faiss)"
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────

st.title("🌐 BGP Troubleshooting RAG Chatbot")
st.markdown(
    "**Powered by Groq (Llama 3.1) + FAISS + Cisco BGP Knowledge Base** "
    "| Deployed on HuggingFace Spaces  \n"
    "Ask any BGP troubleshooting question — answers grounded in Cisco documentation and real telemetry logs."
)
st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Sources retrieved"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

# Handle example question button clicks (populates input via session state)
pending = st.session_state.pop("pending_question", None)

# Chat input
user_input = st.chat_input("Ask a BGP troubleshooting question...")

# Use the pending example question if a button was clicked, otherwise use typed input
question = pending or user_input

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base and querying Groq..."):
            try:
                result = query_bgp(question)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("📄 Sources retrieved"):
                        for src in sources:
                            st.markdown(f"- `{src}`")

                # Save to history with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })

    # Re-run to clear the pending question state cleanly
    if pending:
        st.rerun()
