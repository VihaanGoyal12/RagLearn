from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    # Splits large documents into smaller text chunks for vector embedding
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    raw_chunks = splitter.split_documents(docs)

    # Prefix each chunk with its source and page metadata
    # This allows vector search to match queries like "What is in page 10 of DSnew_merged"
    enhanced_chunks = []
    for chunk in raw_chunks:
        if not chunk.page_content or not chunk.page_content.strip():
            continue
        src = chunk.metadata.get("source_name", chunk.metadata.get("source", "Document"))
        page = chunk.metadata.get("page", 1)
        header = f"[Document: {src} | Page: {page}]\n"

        if not chunk.page_content.startswith("[Document:"):
            chunk.page_content = header + chunk.page_content

        enhanced_chunks.append(chunk)

    print(f"[Chunker] Split {len(docs)} document pages into {len(enhanced_chunks)} chunks.")
    return enhanced_chunks
