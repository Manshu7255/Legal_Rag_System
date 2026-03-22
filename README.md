# ⚖️ Legal RAG System

A Retrieval-Augmented Generation (RAG) system built for answering legal questions — specifically around Indian Supreme Court judgments.

Drop in your PDF case files, ask a question in plain English, and get back a well-structured, citation-backed legal answer. No hallucinations, no guesswork — just facts grounded in the documents you provide.

---

## 💡 What It Does

This system takes a legal query and runs it through a **6-agent pipeline** to produce a reliable, structured response:

1. **Query Analysis** — Understands what you're actually asking
2. **Research Planning** — Figures out what to look for in the documents
3. **Retrieval** — Finds the most relevant chunks from your ingested PDFs using semantic search
4. **Cross-Verification** — Double-checks facts across multiple sources
5. **Hallucination Guard** — Makes sure the answer is actually supported by the retrieved text
6. **Response Formatter** — Packages everything into a clean, structured JSON response

The whole thing is exposed as a simple Flask API — send a POST request with your question, get back a grounded legal answer.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) — runs locally, no API limits |
| Vector DB | ChromaDB (persistent, local storage) |
| PDF Parsing | PyMuPDF |
| API | Flask + Flask-CORS |

---

## 📁 Project Structure

```
legal_rag/
├── app.py                  # Flask API — the main entry point
├── config.py               # All configuration (loaded from .env)
├── ingest.py               # PDF ingestion pipeline (PDF → chunks → embeddings → ChromaDB)
├── requirements.txt        # Python dependencies
├── agents/
│   ├── query_analysis_agent.py
│   ├── research_planning_agent.py
│   ├── retrieval_agent.py
│   ├── cross_verification_agent.py
│   ├── hallucination_guard_agent.py
│   └── response_formatter_agent.py
└── utils/
    ├── pdf_parser.py       # Extracts text from PDFs
    ├── chunker.py          # Splits text into overlapping chunks
    └── schema_builder.py   # Builds the structured response schema
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Manshu7255/Legal_Rag_System.git
cd Legal_Rag_System
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional — these all have sensible defaults
GEMINI_LLM_MODEL=gemini-2.0-flash
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
DATASET_PDF_PATH=./dataset
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K_RESULTS=8
FLASK_PORT=5000
```

> You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Add your PDFs

Drop your Supreme Court judgment PDFs into the `./dataset` folder (or wherever `DATASET_PDF_PATH` points to).

### 6. Ingest the documents

```bash
# Ingest all PDFs
python ingest.py

# Or test with a small sample first
python ingest.py --sample 5

# To start fresh (clear DB and re-ingest)
python ingest.py --reset
```

The first run will download the embedding model (~80MB). After that, it's cached locally.

### 7. Start the API

```bash
python app.py
```

The server will start at `http://localhost:5000`.

---

## 📡 API Endpoints

### `POST /api/query`

Send a legal question, get a structured answer.

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the grounds for divorce under Hindu Marriage Act?"}'
```

### `GET /api/health`

Check if the API and vector DB are ready.

```bash
curl http://localhost:5000/api/health
```

### `GET /api/stats`

Get detailed stats about the vector database — total chunks, collection info, sample data.

```bash
curl http://localhost:5000/api/stats
```

---

## 📝 Notes

- **Embeddings run locally** — no API calls for embedding, so you won't hit any rate limits during ingestion or retrieval.
- **Ingestion is resumable** — if it crashes or you stop it, just run `ingest.py` again. It'll skip files that are already ingested.
- **The `.env` file is gitignored** — your API keys stay safe.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. If you find a bug or have an idea for improvement, I'd love to hear about it.

---

## 📄 License

This project is open source. Use it, modify it, learn from it — just don't use it as actual legal advice. 😄
