from typing import List
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


def create_retriever(
    vectorstore: Chroma,
    docs: List[Document],
    search_k: int = 25,
    reranker_top_n: int = 5,
    mmr_lambda: float = 0.5,
    weights: List[float] = [0.5, 0.5],
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> ContextualCompressionRetriever:
    """
    Creates a hybrid retriever (dense MMR + sparse BM25) with reranking.
    """

    if not isinstance(vectorstore, Chroma):
        raise ValueError("The provided vectorstore must be a Chroma instance.")

    if not docs or len(docs) == 0:
        raise ValueError("No documents provided for BM25 retriever. Did chunking fail?")

    print("🔍 Initializing hybrid retriever (dense MMR + sparse BM25) with reranking...")
    print(f"   • Loaded {len(docs)} docs for BM25")

    # 1. Dense retriever with MMR
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": search_k, "lambda_mult": mmr_lambda},
    )

    # 2. Sparse retriever (BM25)
    try:
        sparse_retriever = BM25Retriever.from_documents(docs)
        sparse_retriever.k = search_k
        print("   • BM25 retriever initialized")
    except Exception as e:
        raise RuntimeError(f"BM25 retriever failed: {e}")

    # 3. Hybrid retriever (dense + sparse)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=weights,
    )

    # 4. Cross-Encoder reranker
    cross_encoder = HuggingFaceCrossEncoder(model_name=cross_encoder_model)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=reranker_top_n)

    # 5. Wrap everything
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=reranker,
    )

    print("✅ Hybrid retriever (MMR + BM25 + reranker) initialized successfully.")
    return compression_retriever
