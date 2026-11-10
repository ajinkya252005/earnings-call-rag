# src/engine.py
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from src.indexer import SimpleHybridRetriever

class EarningsQueryEngine:
    def __init__(self, documents: List[Document], hybrid_retriever: SimpleHybridRetriever):
        self.documents = documents
        self.retriever = hybrid_retriever
        # Lightweight CPU reranker
        self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="data/models")

    def filter_by_quarter(self, quarter: str, year: Optional[int] = None) -> List[Document]:
        """Filters in-memory document pool by exact temporal metadata."""
        filtered = []
        for doc in self.documents:
            match_q = doc.metadata.get("fiscal_quarter", "").upper() == quarter.upper()
            match_y = (doc.metadata.get("fiscal_year") == year) if year else True
            if match_q and match_y:
                filtered.append(doc)
        return filtered

    def rerank_documents(self, query: str, docs: List[Document], top_n: int = 2) -> List[Document]:
        """Reranks retrieved candidate documents using a cross-encoder model."""
        if not docs:
            return []
        
        passages = [
            {"id": idx, "text": d.page_content, "meta": d.metadata}
            for idx, d in enumerate(docs)
        ]
        
        rerank_req = RerankRequest(query=query, passages=passages)
        results = self.reranker.rerank(rerank_req)
        
        # Reconstruct Document instances in reranked order
        ranked_docs = []
        for hit in results[:top_n]:
            doc_idx = hit["id"]
            ranked_docs.append(docs[doc_idx])
            
        return ranked_docs

    def comparative_retrieve(self, query: str, quarters: List[str], top_k_per_quarter: int = 2) -> Dict[str, List[Document]]:
        """
        Guarantees balanced retrieval across target quarters:
        Retrieves candidates independently per quarter, then cross-encoder reranks each set.
        """
        comparative_results = {}

        for q in quarters:
            # 1. Fetch raw candidates matching the specific quarter
            quarter_docs = self.filter_by_quarter(quarter=q)
            
            # Temporary BM25/Dense hybrid on this specific quarter's pool
            # Or run retrieval over full pool and filter down candidates
            all_hits = self.retriever.invoke(query, top_k=20)
            quarter_hits = [d for d in all_hits if d.metadata.get("fiscal_quarter", "").upper() == q.upper()]
            
            # Fallback: if general search yielded few quarter hits, search quarter documents directly
            if len(quarter_hits) < top_k_per_quarter:
                quarter_hits = [d for d in quarter_docs if any(term.lower() in d.page_content.lower() for term in query.split())]
                if not quarter_hits:
                    quarter_hits = quarter_docs[:5]

            # 2. Rerank the quarter candidates
            ranked_quarter_docs = self.rerank_documents(query, quarter_hits, top_n=top_k_per_quarter)
            comparative_results[q] = ranked_quarter_docs

        return comparative_results