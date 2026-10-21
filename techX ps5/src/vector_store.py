import os
import warnings
warnings.filterwarnings("ignore")

import chromadb
from langchain_community.vectorstores import Chroma

class VectorStoreManager:
    # Manages document embeddings using LangChain Chroma vectorstore
    def __init__(self, persist_directory="data/chroma_db", collection_name="langchain_rag"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)
        # Use a single persistent client to avoid SQLite file lock collisions
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_db = None

    def build_from_chunks(self, chunks, embedding_model):
        if not chunks:
            print("[VectorStore] No chunks provided.")
            return None

        # Reset collection cleanly via Chroma client without deleting files on disk
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        # Clean metadata and filter out empty chunks
        valid_chunks = []
        for chunk in chunks:
            if not chunk.page_content or not chunk.page_content.strip():
                continue
            cleaned_meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_meta[k] = v
                else:
                    cleaned_meta[k] = str(v)
            chunk.metadata = cleaned_meta
            valid_chunks.append(chunk)

        print(f"[VectorStore] Indexing {len(valid_chunks)} chunks into LangChain Chroma...")
        self.vector_db = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_model
        )

        # Batch adding to ensure smooth execution
        batch_size = 100
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
            self.vector_db.add_documents(batch)

        print(f"[VectorStore] Successfully stored {self.count()} chunks in Chroma.")
        return self.vector_db

    def load_existing(self, embedding_model):
        try:
            self.vector_db = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=embedding_model
            )
            return self.vector_db
        except Exception as err:
            print(f"[VectorStore] Could not load existing Chroma database: {err}")
            self.vector_db = None
            return None

    def search_similar(self, query, top_k=3):
        if not self.vector_db or self.count() == 0:
            return []

        search_k = min(top_k, self.count())
        try:
            results = self.vector_db.similarity_search_with_relevance_scores(query, k=search_k)
        except Exception:
            results = [(doc, 1.0) for doc in self.vector_db.similarity_search(query, k=search_k)]

        retrieved = []
        for i, (doc, score) in enumerate(results):
            score_val = round(float(score), 4) if score is not None else 1.0
            retrieved.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity": score_val,
                "rank": i + 1
            })
        return retrieved

    def count(self):
        if self.vector_db:
            try:
                return self.vector_db._collection.count()
            except Exception:
                pass
        try:
            col = self.client.get_collection(self.collection_name)
            return col.count()
        except Exception:
            return 0

    def clear(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.vector_db = None
        print("[VectorStore] Database cleared.")
