import pickle
import os
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from app.services.ingestion import bm25_store_path

def get_hybrid_retriever():
    """
    Constructs an Ensemble Retriever combining BM25 (Keywords) and Chroma (Semantic).
    """
    # 1. Load Vector Retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=settings.VECTOR_DB_PATH, 
        embedding_function=embeddings
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Load BM25 Retriever
    if not os.path.exists(bm25_store_path):
        raise ValueError("BM25 index not found. Run ingestion first.")
        
    with open(bm25_store_path, "rb") as f:
        bm25_retriever = pickle.load(f)

    # 3. Combine (Hybrid Search)
    # Weights: 0.5 for Keyword, 0.5 for Semantic
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever