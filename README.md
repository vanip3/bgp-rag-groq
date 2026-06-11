---
title: BGP Troubleshooting RAG Chatbot
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# BGP Troubleshooting RAG Chatbot — Cloud Deployment

Domain-specific RAG chatbot for Cisco BGP troubleshooting, deployed on HuggingFace Spaces with Groq cloud LLM inference. No local setup required — visit the URL and start asking questions.

🔗 **Live Demo:** [huggingface.co/spaces/VP21/bgp-rag-groq](https://huggingface.co/spaces/VP21/bgp-rag-groq)  
📁 **Local Version (nettune):** [github.com/vanip3/bgp-rag-chatbot](https://github.com/vanip3/bgp-rag-chatbot)

---

## Architecture

```
User (Browser)
     │
     ▼
HuggingFace Spaces
(Streamlit App — app.py)
     │
     ├──► FAISS Vector Store ◄── HuggingFace Embeddings
     │    (Pre-built index,        (all-MiniLM-L6-v2)
     │     committed to repo)
     │
     ├──► Top-4 Relevant Chunks
     │
     ▼
Groq Cloud API
(llama-3.1-8b-instant)
     │
     ▼
Answer + Source Documents
```

The user's question is embedded locally using `all-MiniLM-L6-v2`, compared against the pre-built FAISS index to retrieve the 4 most relevant document chunks, then passed to Llama 3.1 8B via the Groq API for answer generation. Source attribution shows which documents informed each answer.

---

## Local vs Cloud Version Comparison

|                  | Local (Project 2)               | Cloud (This Project)         |
|------------------|---------------------------------|------------------------------|
| **LLM**          | nettune (fine-tuned, Ollama)    | Llama 3.1 8B (Groq API)      |
| **Embeddings**   | all-MiniLM-L6-v2                | all-MiniLM-L6-v2             |
| **Vector Store** | FAISS (built at runtime)        | FAISS (pre-built index)      |
| **Access**       | Local only                      | Public URL                   |
| **Cost**         | Free (local GPU/CPU)            | Free (Groq free tier)        |
| **Setup needed** | Ollama + model download         | None — visit URL             |

---

## Key Concepts Demonstrated

**RAG pipeline cloud deployment** — the same retrieval-augmented generation pattern from the local version runs on a public URL with zero infrastructure management. HuggingFace Spaces handles the server, networking, and TLS.

**Secure API key management** — `GROQ_API_KEY` is stored in HF Spaces Secrets (encrypted at rest, injected as an environment variable at runtime). It never appears in code or git history. This is the same pattern used in AWS Secrets Manager, GCP Secret Manager, and Azure Key Vault.

**Pre-built vector index strategy** — the FAISS index is built once locally (`python ingest.py`) and committed to the repo as binary files (`faiss_index/index.faiss`, `faiss_index/index.pkl`). The Space loads these files in ~3 seconds instead of spending 3-5 minutes building the index on every cold start — a critical optimization for serverless deployments with CPU limits.

**Cloud LLM inference via Groq API** — Groq's LPU (Language Processing Unit) hardware delivers ~500 tokens/second on Llama 3.1 8B, free tier, no credit card required. The API is OpenAI-compatible, making it a drop-in for any LangChain LLM component.

**Production Streamlit deployment** — explicit API key validation at startup, graceful error messaging for operators who fork without setting secrets, persistent chat history via `st.session_state`, and source document attribution on every response.

---

## Knowledge Base

Six documents indexed from Cisco BGP documentation and production telemetry:

| File | Content |
|------|---------|
| `bgp_states_guide.txt` | BGP FSM states (IDLE → ESTABLISHED), causes, and verification commands |
| `bgp_troubleshooting_guide.txt` | Systematic methodology for session failures, flapping, route issues |
| `bgp_show_commands.txt` | Complete `show` command reference with output interpretation |
| `bgp_error_messages.txt` | Syslog message reference with NOTIFICATION error codes and remediation |
| `bgp_configuration_guide.txt` | iBGP, Route Reflectors, authentication, filtering, timers |
| `bgp_telemetry_logs.txt` | 30-day production syslog data with incident analysis |

---

## How to Deploy Your Own Copy

1. Fork this repo (or create a new HuggingFace Space and push directly)
2. Create a new HuggingFace Space — select **Streamlit** as the SDK
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME
   git push space main
   ```
4. In Space Settings → **Variables and Secrets**, add:
   - **Name:** `GROQ_API_KEY`
   - **Value:** your key from [console.groq.com](https://console.groq.com)
5. The Space auto-builds and deploys — your public URL is live in ~2 minutes

---

## Local Development

```bash
# 1. Clone
git clone https://github.com/vanip3/bgp-rag-groq
cd bgp-rag-groq

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build FAISS index (one-time)
python ingest.py

# 4. Set API key
export GROQ_API_KEY=gsk_...         # Linux/macOS
# or: set GROQ_API_KEY=gsk_...      # Windows CMD

# 5. Run the app
streamlit run app.py
```

To verify the RAG chain works before launching the UI:
```bash
python rag_chain.py
```

---

## File Structure

```
bgp-rag-groq/
├── app.py                          # Streamlit UI (HF Spaces entry point)
├── rag_chain.py                    # RAG chain: Groq LLM + FAISS retrieval
├── ingest.py                       # Build FAISS index (run once locally)
├── requirements.txt                # Auto-installed by HF Spaces
├── data/
│   ├── docs/
│   │   ├── bgp_states_guide.txt
│   │   ├── bgp_troubleshooting_guide.txt
│   │   ├── bgp_show_commands.txt
│   │   ├── bgp_error_messages.txt
│   │   └── bgp_configuration_guide.txt
│   └── telemetry/
│       └── bgp_telemetry_logs.txt
└── faiss_index/                    # Pre-built index — committed to repo
    ├── index.faiss
    └── index.pkl
```

---

## Tech Stack

- **[LangChain](https://langchain.com)** — RAG chain orchestration (`RetrievalQA`, `PromptTemplate`)
- **[Groq API](https://groq.com)** — Cloud LLM inference (`llama-3.1-8b-instant`, free tier)
- **[HuggingFace Embeddings](https://huggingface.co)** — `sentence-transformers/all-MiniLM-L6-v2` (local, CPU)
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search (Facebook AI Research)
- **[Streamlit](https://streamlit.io)** — Web UI (auto-detected by HF Spaces)
- **[HuggingFace Spaces](https://huggingface.co/spaces)** — Free cloud deployment platform
