from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
import os

def _format_docs(docs: list[Document]) -> str:
    """
    Formats the retrieved documents into a plain string 
    without citations or metadata.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(
    retriever: BaseRetriever, 
    model_name: str = "llama-3.1-8b-instant", 
    temperature: float = 0.0
):
    """
    Creates the full RAG chain for question answering using Groq.
    """
    # 1. Groq LLM
    llm = ChatGroq(
        model_name=model_name, 
        temperature=temperature
    )

    # 2. Prompt template for RAG (strict extraction, no citations!)
    prompt_template = """
    You are a helpful assistant. 
    Use ONLY the provided context to answer the question.
    Do not use outside knowledge.

    - If the context contains a list (bullet points or numbered items),
      reproduce ALL of them exactly as written.
    - Keep the same bullet/number symbols (e.g., •, -, , 1., 2.).
    - Preserve the original line breaks and spacing between items.
    - Do NOT merge list items into one line.
    - If the answer is not explicitly in the context, respond exactly with:
      "The answer is not in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 3. RAG pipeline with LCEL
    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
