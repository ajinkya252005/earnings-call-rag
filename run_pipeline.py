# run_pipeline.py
import os
import sys
from src.parser import load_transcript
from src.indexer import build_hybrid_retriever
from src.engine import EarningsQueryEngine
from src.synthesizer import EarningsSynthesizer

def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please export or set your GOOGLE_API_KEY before running.")
        sys.exit(1)

    print("Step 1: Loading raw transcripts...")
    files = [
        ("data/raw/MSFT_FY26_Q1.txt", "MSFT", 2026, "Q1"),
        ("data/raw/MSFT_FY26_Q2.txt", "MSFT", 2026, "Q2"),
    ]
    
    all_docs = []
    for path, ticker, year, quarter in files:
        docs = load_transcript(path, ticker, year, quarter)
        all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} total chunks across Q1 and Q2.")

    print("\nStep 2: Building hybrid indices (Dense + Sparse BM25)...")
    hybrid_retriever = build_hybrid_retriever(all_docs, top_k=8)

    print("\nStep 3: Initializing Partitioned Query Engine & FlashRank Reranker...")
    engine = EarningsQueryEngine(documents=all_docs, hybrid_retriever=hybrid_retriever)
    
    print("\nStep 4: Initializing Gemini Synthesizer...")
    synthesizer = EarningsSynthesizer(model_name="gemini-3.5-flash", temperature=0.0)

    # Comparative Guidance Query
    query = "How did Microsoft's CapEx guidance and cloud capacity constraints evolve from Q1 to Q2?"
    quarters = ["Q1", "Q2"]

    print(f"\nStep 5: Running partitioned comparative retrieval across {quarters}...")
    comparative_results = engine.comparative_retrieve(query, quarters=quarters, top_k_per_quarter=2)

    print("\nStep 6: Generating Grounded Synthesis via Gemini...")
    analysis = synthesizer.analyze(query, comparative_results)

    print("\n" + "=" * 80)
    print("FINANCIAL DELTA & EVOLUTION ANALYSIS (GEMINI)")
    print("=" * 80)
    print(analysis)

if __name__ == "__main__":
    main()