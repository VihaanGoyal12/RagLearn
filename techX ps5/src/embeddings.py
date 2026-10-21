import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    # Load HuggingFace embeddings via LangChain
    print(f"[Embedder] Loading LangChain HuggingFaceEmbeddings: {model_name}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("[Embedder] Ready!")
    return embedding_model
