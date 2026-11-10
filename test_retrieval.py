# test_retrieval.py
from src.parser import load_transcript
from src.indexer import build_hybrid_retriever

def main():
    files = [
        ("data/raw/MSFT_FY26_Q1.txt", "MSFT", 2026, "Q1"),
        ("data/raw/MSFT_FY26_Q2.txt", "MSFT", 2026, "Q2"),
    ]
    
    all_docs = []
    for path, ticker, year, quarter in files:
        docs = load_transcript(path, ticker, year, quarter)
        all_docs.extend(docs)
        
    print(f"Loaded {len(all_docs)} total chunks.")
    
    retriever = build_hybrid_retriever(all_docs, top_k=4)
    
    test_queries = [
        "What was Azure and cloud revenue growth percentage?",
        "How are customers approaching AI return on investment and infrastructure capacity?"
    ]
    
    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)
        
        results = retriever.invoke(q, top_k=3)
        for rank, doc in enumerate(results, start=1):
            meta = doc.metadata
            quarter = f"{meta.get('ticker')} {meta.get('fiscal_quarter')} FY{meta.get('fiscal_year')}"
            section = meta.get("section")
            speaker = meta.get("speaker") or f"{meta.get('analyst_name')} -> {meta.get('executive_name')}"
            
            print(f"\n[Hit #{rank}] [{quarter} | {section.upper()} | Speaker: {speaker}]")
            snippet = doc.page_content.replace("\n", " ")[:200]
            print(f"Content: {snippet}...")

if __name__ == "__main__":
    main()