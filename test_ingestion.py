# test_ingestion.py
from src.parser import load_transcript

def run_test():
    files = [
        ("data/raw/MSFT_FY26_Q1.txt", "MSFT", 2026, "Q1"),
        ("data/raw/MSFT_FY26_Q2.txt", "MSFT", 2026, "Q2"),
    ]
    
    total_docs = []
    
    for path, ticker, year, quarter in files:
        print(f"--- Loading {path} ---")
        try:
            docs = load_transcript(path, ticker, year, quarter)
            print(f"Generated {len(docs)} LangChain chunks.")
            
            # Count distribution
            remarks_count = sum(1 for d in docs if d.metadata.get("section") == "prepared_remarks")
            qa_count = sum(1 for d in docs if d.metadata.get("section") == "qa")
            print(f"  - Prepared Remarks chunks: {remarks_count}")
            print(f"  - Q&A chunks: {qa_count}")
            
            # Print sample Q&A chunk
            sample_qa = next((d for d in docs if d.metadata.get("section") == "qa"), None)
            if sample_qa:
                print("\n[Sample Q&A Metadata]:", sample_qa.metadata)
                print("[Sample Q&A Content Preview]:\n", sample_qa.page_content[:250], "\n...\n")
                
            total_docs.extend(docs)
        except FileNotFoundError:
            print(f"File not found: {path}. Check file names in data/raw/")
            
    print(f"Total chunks ready for vector index across both quarters: {len(total_docs)}")

if __name__ == "__main__":
    run_test()