import streamlit as st
from dotenv import load_dotenv
import os
import uuid

from helper.file_loader import load_pdf
from helper.chunking import chunk_documents
from helper.embed_vectorstore import create_vector_store, load_vector_store
from helper.retriever_reranker import create_retriever
from helper.chainer import create_rag_chain

# --- App Configuration ---
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("📄 Ask Questions to Your PDF 💬")

# Load environment variables (e.g., for API keys)
load_dotenv()

# Directory for persisted vector store
DB_DIR = "db"

# --- Initialize session state ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for PDF upload ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Process PDF"):
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                temp_path = None
                try:
                    # Save PDF temporarily with unique filename
                    os.makedirs("temp", exist_ok=True)
                    temp_path = os.path.join("temp", f"{uuid.uuid4()}_{uploaded_file.name}")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    # --- Load and chunk PDF ---
                    st.write("📥 Debug: Loading PDF...")
                    docs = load_pdf(temp_path)
                    st.write(f"📄 Debug: Loaded {len(docs)} pages.")
                    print(f"DEBUG: Loaded {len(docs)} pages from {temp_path}")

                    if not docs:
                        raise ValueError("load_pdf returned no documents!")

                    st.write("✂️ Debug: Chunking documents...")
                    chunks = chunk_documents(docs)
                    st.write(f"🧩 Debug: Created {len(chunks)} chunks.")
                    print(f"DEBUG: Created {len(chunks)} chunks from {len(docs)} pages")

                    if not chunks:
                        raise ValueError("chunk_documents returned no chunks!")

                    # --- Try to load existing vector store ---
                    vector_store = load_vector_store(DB_DIR)

                    if vector_store:
                        st.info("♻️ Using existing persisted vector store.")
                    else:
                        st.info("⚡ Creating new vector store (this may take a while)...")
                        vector_store = create_vector_store(chunks, persist=True, persist_dir=DB_DIR)
                        st.success(f"✅ Debug: Vector store created with {len(chunks)} chunks.")

                    # Create hybrid retriever (with reranker + MMR)
                    retriever = create_retriever(
                        vectorstore=vector_store, # Dense retriever uses embeddings from here
                        docs=chunks,  # Always pass chunks so BM25 works
                        search_k=15,
                        reranker_top_n=5,
                        mmr_lambda=0.5
                    )

                    # Create RAG chain
                    rag_chain = create_rag_chain(retriever)

                    # Store in session state
                    st.session_state.retriever = retriever
                    st.session_state.rag_chain = rag_chain
                    st.session_state.processed_file = uploaded_file.name

                    st.success(f"✅ {uploaded_file.name} processed successfully!")

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    st.exception(e)  # shows full traceback in dev mode
                finally:
                    # Clean up temp file
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.warning("Please upload a PDF before clicking Process.")

# --- Main area for Q&A ---
st.header("Q&A")

if st.session_state.rag_chain:
    st.info(f"Ready to answer questions from **{st.session_state.processed_file}**")
    user_query = st.chat_input("Ask a question about your PDF...")

    if user_query:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("Generating answer..."):
            try:
                response = st.session_state.rag_chain.invoke(user_query)

                # Handle both dict and string outputs
                if isinstance(response, dict):
                    answer = response.get("answer") or response.get("result") or str(response)
                else:
                    answer = str(response)

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.exception(e)

    # --- Display chat messages ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
else:
    st.info("📥 Upload and process a PDF from the sidebar to begin.")
