import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os

# Directory to persist FAISS index
PERSIST_DIR = "faiss_db"

# Singleton embeddings instance (cached for performance)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_store(
    docs: List[Document], 
    persist: bool = True, 
    persist_dir: str = PERSIST_DIR
) -> FAISS:
    """
    Creates a FAISS vector store from a list of documents.

    Args:
        docs (List[Document]): List of chunked documents.
        persist (bool): Whether to save the FAISS index to disk.
        persist_dir (str): Directory for persistence.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    if not docs:
        raise ValueError("No documents provided for creating the vector store.")

    embedding_model = get_embeddings()

    # Build FAISS index from documents
    vectorstore = FAISS.from_documents(docs, embedding_model)

    if persist:
        os.makedirs(persist_dir, exist_ok=True)
        vectorstore.save_local(persist_dir)
        print(f"✅ FAISS vector store created and saved to {persist_dir}")

    return vectorstore

def load_vector_store(
    persist_dir: str = PERSIST_DIR
) -> Optional[FAISS]:
    """
    Loads an existing persisted FAISS vector store.
    Returns None if not found.
    """
    if not os.path.exists(persist_dir):
        st.warning("⚠️ No persisted FAISS vector store found.")
        return None

    try:
        embedding_model = get_embeddings()
        return FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Failed to load FAISS vector store: {e}")
        return None
