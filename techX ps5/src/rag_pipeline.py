import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class LangChainRAGPipeline:
    # Full LangChain retrieval-augmented generation pipeline using Groq LLM
    def __init__(self, vector_store_manager, embedding_model, groq_api_key=None, model_name=None):
        self.store = vector_store_manager
        self.embedding_model = embedding_model

        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing! Please configure it in .env")

        self.model_name = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        # Initialize LangChain ChatGroq model
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.2,
            max_tokens=1024
        )
        print(f"[RAGPipeline] Initialized with LangChain ChatGroq: {self.model_name}")

        # LangChain Prompt Template
        self.prompt = PromptTemplate(
            template="""You are a helpful and knowledgeable AI assistant.
Answer the user's question accurately using ONLY the provided document context below.
If the answer cannot be found in the context, truthfully state that the documents do not contain enough information.

Document Context:
{context}

User Question:
{question}

Answer:""",
            input_variables=["context", "question"]
        )

        # Build LangChain LCEL chain: prompt | llm | output_parser
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

    def ask(self, query, top_k=3):
        # 1. Retrieve matching chunks from LangChain Chroma
        retrieved_docs = self.store.search_similar(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "The database is empty or no relevant documents were found. Please upload and index documents first.",
                "sources": [],
                "context_used": ""
            }

        # 2. Build combined context
        context_parts = []
        for i, item in enumerate(retrieved_docs):
            src = item["metadata"].get("source_name", item["metadata"].get("source", "Document"))
            page = item["metadata"].get("page", 1)
            context_parts.append(f"[Source {i+1}: {src} (Page {page})]\n{item['text']}")

        combined_context = "\n\n".join(context_parts)

        # 3. Run the LangChain chain
        try:
            answer = self.rag_chain.invoke({
                "context": combined_context,
                "question": query
            })
        except Exception as err:
            answer = f"Error generating answer with Groq LLM: {err}"

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "context_used": combined_context
        }
