import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

def load_pdf_file(file_path):
    # Try standard PDF text extraction first
    docs = []
    try:
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()
    except Exception:
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        except Exception:
            docs = []

    file_name = Path(file_path).name

    # Check if the PDF has actual text or if it's a scanned/image-only document
    total_text_len = sum(len(d.page_content.strip()) for d in docs)
    if total_text_len < 50:
        print(f"[Loader] '{file_name}' contains scanned images (no digital text layer). Running OCR...")
        try:
            import fitz
            from rapidocr_onnxruntime import RapidOCR

            ocr = RapidOCR()
            doc_fitz = fitz.open(str(file_path))
            ocr_docs = []
            for page_idx, page in enumerate(doc_fitz):
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                result, _ = ocr(img_bytes)
                page_text = "\n".join([line[1] for line in result]) if result else ""

                doc_item = Document(
                    page_content=page_text,
                    metadata={
                        "source_name": file_name,
                        "file_type": "pdf",
                        "page": page_idx + 1,
                        "total_pages": len(doc_fitz)
                    }
                )
                ocr_docs.append(doc_item)

            print(f"[Loader] OCR completed for {file_name} ({len(ocr_docs)} pages).")
            return ocr_docs
        except Exception as ocr_err:
            print(f"[Loader] OCR fallback failed for {file_name}: {ocr_err}")

    # Standard PDF text metadata formatting
    for idx, doc in enumerate(docs):
        doc.metadata["source_name"] = file_name
        doc.metadata["file_type"] = "pdf"
        # Make page numbers 1-indexed for human readability
        current_page = doc.metadata.get("page", idx)
        if isinstance(current_page, int):
            doc.metadata["page"] = current_page + 1

    return docs

def load_text_file(file_path):
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    file_name = Path(file_path).name
    for doc in docs:
        doc.metadata["source_name"] = file_name
        doc.metadata["file_type"] = "txt"
        doc.metadata["page"] = 1
    return docs

def load_all_documents(folder_path):
    all_docs = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"[Loader] Folder {folder_path} does not exist!")
        return all_docs

    files = list(folder.glob("**/*"))
    print(f"[Loader] Found {len(files)} total files in {folder_path}")

    for file_path in files:
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                try:
                    pdf_docs = load_pdf_file(file_path)
                    all_docs.extend(pdf_docs)
                    print(f"  ✓ Loaded PDF: {file_path.name} ({len(pdf_docs)} pages)")
                except Exception as err:
                    print(f"  ✗ Failed loading {file_path.name}: {err}")
            elif ext == ".txt":
                try:
                    txt_docs = load_text_file(file_path)
                    all_docs.extend(txt_docs)
                    print(f"  ✓ Loaded Text: {file_path.name} ({len(txt_docs)} docs)")
                except Exception as err:
                    print(f"  ✗ Failed loading {file_path.name}: {err}")

    print(f"[Loader] Finished! Total loaded pages/docs: {len(all_docs)}")
    return all_docs
