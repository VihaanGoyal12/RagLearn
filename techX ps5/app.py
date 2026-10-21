import os
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' resolves to techX ps5/src regardless of execution directory
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.document_loader import load_all_documents  # pyrefly: ignore[missing-import]
from src.text_splitter import split_documents      # pyrefly: ignore[missing-import]
from src.embeddings import get_embedding_model     # pyrefly: ignore[missing-import]
from src.vector_store import VectorStoreManager    # pyrefly: ignore[missing-import]
from src.rag_pipeline import LangChainRAGPipeline  # pyrefly: ignore[missing-import]

load_dotenv()

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
STATIC_DIR = BASE_DIR / "static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize core LangChain RAG components
print("[App] Initializing LangChain components...")
embedder = get_embedding_model(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
vector_store = VectorStoreManager(persist_directory=str(CHROMA_DIR), collection_name="langchain_rag")
vector_store.load_existing(embedder)
rag_pipeline = LangChainRAGPipeline(vector_store_manager=vector_store, embedding_model=embedder)

app = FastAPI(title="TechX PS5 - AI-Powered RAG System", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    model_name: str = "openai/gpt-oss-20b"

class IndexRequest(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200

@app.get("/")
def serve_home():
    # Serves the main UI
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "TechX PS5 RAG Backend API is running."}

@app.get("/api/status")
def get_system_status():
    uploaded_files = list(UPLOAD_DIR.glob("*"))
    files_info = [f.name for f in uploaded_files if f.is_file()]
    groq_key_set = bool(os.getenv("GROQ_API_KEY"))
    
    return {
        "status": "online",
        "indexed_chunks": vector_store.count(),
        "uploaded_files_count": len(files_info),
        "files": files_info,
        "embedding_model": embedder.model_name,
        "llm_model": rag_pipeline.model_name,
        "groq_configured": groq_key_set
    }

@app.get("/api/documents")
def list_uploaded_documents():
    docs_list = []
    if UPLOAD_DIR.exists():
        for f in sorted(UPLOAD_DIR.glob("*")):
            if f.is_file():
                docs_list.append({
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "type": f.suffix.lower().replace(".", "")
                })
    return {"documents": docs_list}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename).name
    target_path = UPLOAD_DIR / filename
    
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"success": True, "filename": filename, "message": f"Uploaded '{filename}' successfully."}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {err}")

@app.post("/api/delete_document")
def delete_document(filename: str = Form(...)):
    target_path = UPLOAD_DIR / Path(filename).name
    if target_path.exists() and target_path.is_file():
        os.remove(target_path)
        return {"success": True, "message": f"Deleted '{filename}'"}
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/index")
def index_documents(req: IndexRequest = IndexRequest()):
    # 1. Document Ingestion
    start_time = time.time()
    docs = load_all_documents(str(UPLOAD_DIR))
    
    if not docs:
        return {
            "success": False,
            "message": "No documents found in uploads folder to index.",
            "chunk_count": 0
        }

    # 2. Chunking Strategy
    chunks = split_documents(docs, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    
    # 3 & 4. LangChain Vector Store Ingestion (Chroma.from_documents embeds and persists automatically)
    vector_store.build_from_chunks(chunks, embedding_model=embedder)
    added_count = vector_store.count()
    
    total_time = round(time.time() - start_time, 2)
    return {
        "success": True,
        "message": f"Successfully indexed {added_count} chunks from {len(docs)} document pages.",
        "documents_loaded": len(docs),
        "chunks_indexed": added_count,
        "time_seconds": total_time
    }

@app.post("/api/clear")
def clear_vector_database():
    vector_store.clear()
    return {"success": True, "message": "Vector database has been cleared."}

@app.post("/api/query")
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    
    # Update model if specified
    if req.model_name and req.model_name != rag_pipeline.model_name:
        try:
            rag_pipeline.llm.model_name = req.model_name
            rag_pipeline.model_name = req.model_name
        except Exception:
            pass

    # Ask the pipeline
    result = rag_pipeline.ask(query=req.query, top_k=req.top_k)
    elapsed_time = round(time.time() - start_time, 2)

    return {
        "query": req.query,
        "answer": result["answer"],
        "sources": result["sources"],
        "response_time": elapsed_time,
        "model_used": rag_pipeline.model_name
    }

if __name__ == "__main__":
    import uvicorn
    import socket

    def get_available_port(preferred_port=8000):
        for port in [preferred_port, 8080, 8001, 8002]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return preferred_port

    target_port = int(os.getenv("PORT", get_available_port(8000)))
    print(f"\n🚀 Server starting on: http://127.0.0.1:{target_port}\n")
    uvicorn.run("app:app", host="127.0.0.1", port=target_port, reload=True)
