import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os

# Directory to persist vector DB
PERSIST_DIR = "chroma_db"

# Singleton embeddings instance (cached for performance)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # Device is auto-detected by default
    )

def create_vector_store(
    docs: List[Document], 
    persist: bool = True, 
    persist_dir: str = PERSIST_DIR
) -> Chroma:
    """
    Creates a Chroma vector store from a list of documents.

    Args:
        docs (List[Document]): List of chunked documents.
        persist (bool): Whether to persist the DB to disk.
        persist_dir (str): Directory for persistence.

    Returns:
        Chroma: A Chroma vector store instance.
    """
    if not docs:
        raise ValueError("No documents provided for creating the vector store.")

    embedding_model = get_embeddings()

    if persist:
        os.makedirs(persist_dir, exist_ok=True)
        # Create empty persistent DB, then add documents
        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=persist_dir
        )
        vectorstore.add_documents(docs)
        vectorstore.persist()
        print(f"✅ Vector store created and saved to {persist_dir}")
    else:
        # In-memory only
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

    return vectorstore

def load_vector_store(
    persist_dir: str = PERSIST_DIR
) -> Optional[Chroma]:
    """
    Loads an existing persisted Chroma vector store.
    Returns None if not found.
    """
    if not os.path.exists(persist_dir):
        st.warning("⚠️ No persisted vector store found.")
        return None

    try:
        embedding_model = get_embeddings()
        return Chroma(
            persist_directory=persist_dir, 
            embedding_function=embedding_model
        )
    except Exception as e:
        st.error(f"❌ Failed to load vector store: {e}")
        return None
