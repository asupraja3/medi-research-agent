import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import pickle
import os
from app.core.config import settings

# Global cache for the BM25 retriever (since it's memory-based)
bm25_store_path = os.path.join(settings.VECTOR_DB_PATH, "bm25_retriever.pkl")

def ingest_data():
    """
    Reads the Kaggle CSV, processes it, and stores it in VectorDB and BM25.
    """
    print("--- Starting Ingestion ---")
    
    # 1. Load Data (Simulating Real-time fetch from local file)
    if not os.path.exists(settings.DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {settings.DATA_PATH}")
    
    df = pd.read_csv(settings.DATA_PATH)
    # Filter for valid transcriptions
    df = df.dropna(subset=['transcription'])
    # Combine relevant columns for context
    df['combined_text'] = "Medical Specialty: " + df['medical_specialty'] + \
                          "\nTranscription: " + df['transcription']

    # 2. Convert to LangChain Documents
    loader = DataFrameLoader(df, page_content_column="combined_text")
    docs = loader.load()

    # 3. Split Text (Chunking strategy)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 4. Create Vector Store (Semantic Search)
    # Using ChromaDB locally
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=settings.VECTOR_DB_PATH
    )
    print("Vector Store Updated.")

    # 5. Create BM25 Retriever (Keyword Search)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5  # Retrieve top 5 by keyword
    
    # Save BM25 object to disk (simple persistence for this demo)
    with open(bm25_store_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("BM25 Index Saved.")
    
    return True