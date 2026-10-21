# TechX Problem Statement 5: AI-Powered Backend / RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system built with simple, readable Python modules, Hugging Face sentence embeddings, persistent ChromaDB vector storage, Groq LLM inference, and an interactive web frontend.

---

## 📌 Architecture & Core Pillars

This project implements the 4 key stages required for Problem Statement 5:

1. **Document Ingestion (`src/document_loader.py`)**
   - Loads multiple documents (PDFs and TXT files) from the `data/uploads/` directory.
   - Attaches metadata (source filename, document type, page numbers) to each document.
   - Supports uploading new documents directly through the web frontend.

2. **Chunking Strategy (`src/text_splitter.py`)**
   - Splits documents into manageable text chunks using `RecursiveCharacterTextSplitter`.
   - Uses configurable chunk size (default: 1000 characters) and overlap (default: 200 characters) to preserve semantic context across chunk boundaries.

3. **Embeddings & Vector Search (`src/embeddings.py` & `src/vector_store.py`)**
   - Generates dense vector embeddings using the Hugging Face `all-MiniLM-L6-v2` SentenceTransformer model (384 dimensions).
   - Stores chunks and vector representations into a local persistent ChromaDB collection.
   - Performs cosine similarity search to retrieve the top $K$ most relevant document chunks for any user question.

4. **LLM-Based Response Generation (`src/rag_pipeline.py`)**
   - Combines retrieved document chunks into a structured context prompt.
   - Passes the prompt and question to a Groq LLM (`openai/gpt-oss-20b` or `qwen/qwen3.8-27b`) via the LangChain Groq client.
   - Produces grounded, hallucination-free answers along with citations and similarity scores.

5. **Web Frontend (`static/` & `app.py`)**
   - Clean, modern UI to drag-and-drop upload PDFs, trigger vector indexing, ask questions, and inspect the retrieved source chunks.

---

## 📂 Project Directory Structure

```text
techX ps5/
├── .env                     # API keys and environment variables
├── requirements.txt         # Project dependencies
├── README.md                # Documentation and architecture guide
├── app.py                   # FastAPI backend server & static web app
├── test_rag.py              # CLI test script for pipeline verification
├── data/
│   ├── uploads/             # Raw PDF and TXT documents
│   └── chroma_db/           # Persistent ChromaDB vector database
├── src/
│   ├── __init__.py
│   ├── document_loader.py   # Step 1: Document ingestion
│   ├── text_splitter.py     # Step 2: Text chunking strategy
│   ├── embeddings.py        # Step 3A: Hugging Face SentenceTransformer
│   ├── vector_store.py      # Step 3B: ChromaDB vector search & storage
│   └── rag_pipeline.py      # Step 4: RAG retrieval + Groq LLM generation
└── static/
    ├── index.html           # Frontend HTML structure
    ├── style.css            # Dark-theme styling & responsive layout
    └── app.js               # Frontend JavaScript API interactions
```

---

## 🚀 Getting Started

### 1. Requirements

Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Check `techX ps5/.env` and verify your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
GROQ_MODEL=openai/gpt-oss-20b
```

---

## 💻 Running the Application

### Option A: Launch the Web Frontend (Recommended)

Run the FastAPI application from inside `techX ps5`:

```bash
python app.py
```

or with `uvicorn`:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

#### Web UI Features:
- **Upload Documents**: Drag and drop any `.pdf` or `.txt` files.
- **Index Database**: Click `⚡ Index & Build Vector DB` to chunk and embed all uploaded files.
- **Ask Query**: Type your question and click `Ask AI`.
- **View Citations**: Expand the `Retrieved Context & Sources` dropdown to see the exact text snippets, source files, page numbers, and similarity percentages.

---

### Option B: Run the Command-Line Test Script

To verify document loading, chunking, embedding, vector search, and LLM generation in the terminal without opening a browser:

```bash
python test_rag.py
```

Sample output:
```text
=== Step 1: Loading Documents ===
[Loader] Found 6 total files in data/uploads
  ✓ Loaded PDF: attention.pdf (11 pages)
  ✓ Loaded PDF: embeddings.pdf (27 pages)
  ...
=== Step 2: Splitting into Chunks ===
[Chunker] Split 136 document pages into 249 chunks.

=== Step 3: Generating Embeddings ===
[Embedder] Ready! Embedding dimension: 384

=== Step 4: Storing in ChromaDB ===
[VectorStore] Added 249 chunks. New total count: 249

=== Step 5: Testing RAG Pipeline ===
Query: 'What is the attention mechanism in transformers?'
--- Answer ---
Self-attention is an attention mechanism that relates different positions of a single sequence in order to compute a representation of the sequence...
```

---

## ⚙️ Key Technical Decisions

| Component | Choice | Reason |
| :--- | :--- | :--- |
| **Document Loader** | LangChain `PyMuPDFLoader` / `PyPDFLoader` | Fast extraction of text and metadata across documents |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Preserves paragraphs and natural sentence structure |
| **Embeddings** | LangChain `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) | Embeddings via LangChain interface running locally |
| **Vector DB** | LangChain `Chroma` (Persistent) | Serverless vector database with similarity search & scores |
| **LLM Inference** | LangChain `ChatGroq` (`openai/gpt-oss-20b`) | Ultra-fast token generation speed (< 1s latency) |
| **Prompt & Chain** | LangChain `PromptTemplate` + LCEL (`prompt \| llm`) | Declarative, modern LangChain chain syntax |
| **API & UI** | FastAPI + Vanilla JS/CSS | Zero external frontend build tools needed; works directly out of the box |
