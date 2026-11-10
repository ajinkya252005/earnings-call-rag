# src/indexer.py
import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

CHROMA_PERSIST_DIR = "data/chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

class SimpleHybridRetriever:
    """
    A transparent Hybrid Retriever that combines Dense (Chroma) and Sparse (BM25)
    retrievers using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, dense_retriever, bm25_retriever, dense_weight=0.5, sparse_weight=0.5, rrf_k=60):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def invoke(self, query: str, top_k: int = 4) -> List[Document]:
        # 1. Fetch hits from both systems
        dense_docs = self.dense_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 2. Score with Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Rank dense results
        for rank, doc in enumerate(dense_docs, start=1):
            key = doc.page_content
            doc_map[key] = doc
            score = self.dense_weight / (self.rrf_k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score

        # Rank sparse (BM25) results
        for rank, doc in enumerate(bm25_docs, start=1):
            key = doc.page_content
            doc_map[key] = doc
            score = self.sparse_weight / (self.rrf_k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score

        # 3. Sort by combined RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:top_k]]


def build_hybrid_retriever(
    documents: List[Document], 
    persist_directory: str = CHROMA_PERSIST_DIR,
    top_k: int = 5
) -> SimpleHybridRetriever:
    embeddings = get_embedding_function()
    
    # 1. Dense Chroma Vector Store
    print(f"Indexing {len(documents)} documents into Chroma...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="earnings_transcripts"
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # 2. Sparse BM25 Retriever
    print(f"Indexing {len(documents)} documents into BM25...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k
    
    # 3. Combine with custom RRF
    return SimpleHybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever
    )