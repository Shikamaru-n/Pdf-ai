from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 500
) -> List[Document]:
    """
    Splits PDF pages into smaller text chunks for embedding, 
    while preserving useful metadata like source and page numbers.

    Args:
        documents (List[Document]): List of LangChain Document objects (usually per page)
        chunk_size (int): Max size of each chunk
        chunk_overlap (int): Overlap between chunks to preserve context

    Returns:
        List[Document]: List of chunked Document objects
    """
    if not documents:
        raise ValueError("No documents provided for chunking.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        # No separators → use defaults, avoids over-splitting
    )

    # Perform the split
    chunked_docs = text_splitter.split_documents(documents)

    # Ensure metadata is preserved and normalized
    for i, doc in enumerate(chunked_docs):
        source = doc.metadata.get("source", "unknown")
        page = (
            doc.metadata.get("page") 
            or doc.metadata.get("page_number") 
            or "n/a"
        )

        source_name = os.path.basename(str(source))

        # Use a globally unique chunk_id (source + index)
        doc.metadata.update({
            "chunk_id": f"{source_name}_{i}",
            "source": source_name,
            "page": page
        })

    return chunked_docs
