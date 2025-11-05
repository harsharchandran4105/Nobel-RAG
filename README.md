# ⚡ RAG Quickstart (2-hour build)

Minimal Retrieval-Augmented Generation (RAG) project you can run locally and showcase on GitHub/resume.
- 🔎 Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- 🧠 Vector store: FAISS (local)
- 📄 Ingest: Text/PDF files from `data/`
- ✍️ Generator: OpenAI (if `OPENAI_API_KEY` is set) or fallback to local `google/flan-t5-base`
- 🖥️ UI: Streamlit (`streamlit_app.py`)
- 🌐 API: FastAPI (`app.py`)

## Quick Start (iq200 fast path)
```bash
# 0) Create & activate env (Linux/Mac)
python -m venv .venv && source .venv/bin/activate
# (Windows) .venv\Scripts\activate

# 1) Install deps
pip install -r requirements.txt

# 2) Put PDFs or .txt into ./data/ (sample included)
# 3) Build index
python ingest.py

# 4) Run UI
streamlit run streamlit_app.py
# or run API
uvicorn app:app --reload --port 8000
```

### Example query
- Question: *"What is Retrieval-Augmented Generation?"*
- The app retrieves top-k chunks from your local docs and answers using OpenAI or Flan-T5.

## Project Structure
```text
rag-quickstart/
  app.py               # FastAPI RAG endpoint
  streamlit_app.py     # Simple UI
  ingest.py            # Build FAISS index from data/
  rag.py               # Core RAG components
  data/                # Your .txt/.pdf files
  store/               # Saved FAISS + metadata (auto-created)
  requirements.txt
  .env.example
  README.md
```

## Notes
- For OpenAI generation, copy `.env.example` to `.env` and add your key.
- For offline demo, the fallback `flan-t5-base` runs but is slower/weaker—good enough for a portfolio demo.

## License
MIT