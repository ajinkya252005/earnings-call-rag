# test_comparative.py
from src.parser import load_transcript
from src.indexer import build_hybrid_retriever
from src.engine import EarningsQueryEngine

def main():
    files = [
        ("data/raw/MSFT_FY26_Q1.txt", "MSFT", 2026, "Q1"),
        ("data/raw/MSFT_FY26_Q2.txt", "MSFT", 2026, "Q2"),
    ]
    
    all_docs = []
    for path, ticker, year, quarter in files:
        docs = load_transcript(path, ticker, year, quarter)
        all_docs.extend(docs)
        
    print(f"Loaded {len(all_docs)} total chunks across Q1 and Q2.")
    
    # Build base hybrid index
    hybrid_retriever = build_hybrid_retriever(all_docs, top_k=10)
    
    # Initialize partitioned engine
    engine = EarningsQueryEngine(documents=all_docs, hybrid_retriever=hybrid_retriever)
    
    test_query = "What did management state regarding capital expenditures (CapEx) and cloud capacity?"
    target_quarters = ["Q1", "Q2"]
    
    print("\n" + "=" * 80)
    print(f"COMPARATIVE QUERY: {test_query}")
    print("=" * 80)
    
    results = engine.comparative_retrieve(test_query, quarters=target_quarters, top_k_per_quarter=2)
    
    for q, docs in results.items():
        print(f"\n--- TARGET PERIOD: {q} ({len(docs)} top chunks reranked) ---")
        for idx, doc in enumerate(docs, start=1):
            speaker = doc.metadata.get("speaker") or f"{doc.metadata.get('analyst_name')} -> {doc.metadata.get('executive_name')}"
            section = doc.metadata.get("section", "").upper()
            print(f"\n[Result #{idx} | {section} | {speaker}]")
            print(doc.page_content[:250].replace("\n", " ") + "...\n")

if __name__ == "__main__":
    main()