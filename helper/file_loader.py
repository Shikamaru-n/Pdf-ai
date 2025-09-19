import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Loads a PDF file and returns it as a list of LangChain Document objects (one per page).
    
    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Document]: A list of documents, one for each page of the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file at {pdf_path} does not exist.")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()  # returns List[Document]

        if not documents:
            raise ValueError("No readable text found in the PDF.")

        # Ensure metadata consistency without overwriting existing keys
        for doc in documents:
            doc.metadata.setdefault("source", os.path.basename(pdf_path))

        return documents

    except Exception as e:
        raise RuntimeError(f"Error processing PDF at {pdf_path}") from e
