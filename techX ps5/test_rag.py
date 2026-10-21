import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path so 'src' resolves to techX ps5/src
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.document_loader import load_all_documents  # pyrefly: ignore[missing-import]
from src.text_splitter import split_documents      # pyrefly: ignore[missing-import]
from src.embeddings import get_embedding_model     # pyrefly: ignore[missing-import]
from src.vector_store import VectorStoreManager    # pyrefly: ignore[missing-import]
from src.rag_pipeline import LangChainRAGPipeline  # pyrefly: ignore[missing-import]

def run_test():
    load_dotenv()
    print("=== Step 1: Loading Documents with LangChain ===")
    docs = load_all_documents("data/uploads")
    print(f"Loaded {len(docs)} document pages.")

    print("\n=== Step 2: Chunking with LangChain RecursiveCharacterTextSplitter ===")
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

    print("\n=== Step 3: Initializing LangChain HuggingFaceEmbeddings ===")
    embedder = get_embedding_model(model_name="all-MiniLM-L6-v2")

    print("\n=== Step 4: Storing & Indexing into LangChain Chroma ===")
    store = VectorStoreManager(persist_directory="data/chroma_db", collection_name="langchain_rag")
    store.build_from_chunks(chunks, embedding_model=embedder)

    print("\n=== Step 5: Testing LangChain RAG Pipeline with Groq LLM ===")
    rag = LangChainRAGPipeline(vector_store_manager=store, embedding_model=embedder)

    test_query = "What is the attention mechanism in transformers?"
    print(f"Query: '{test_query}'")
    result = rag.ask(test_query, top_k=2)

    print("\n--- Answer from Groq ---")
    print(result["answer"])

    print("\n--- Sources Retrieved ---")
    for s in result["sources"]:
        src_name = s['metadata'].get('source_name', s['metadata'].get('source', 'Unknown'))
        print(f"- {src_name} (Similarity: {s['similarity']})")

if __name__ == "__main__":
    run_test()
