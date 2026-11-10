# test_router.py
import os
from dotenv import load_dotenv
from src.router import QueryRouter

load_dotenv()

def main():
    router = QueryRouter()
    
    test_queries = [
        "Compare Microsoft's CapEx guidance and GPU capacity allocation between Q1 and Q2.",
        "What did Amy Hood guide for gross margins in Q2 2026?",
        "Are they seeing customer budget delays due to macro headwinds across the first two quarters?",
        "Tell me about the OpenAI contract terms mentioned by Satya."
    ]
    
    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"INPUT: {q}")
        print("=" * 80)
        parsed = router.route(q)
        print(f"Ticker:                 {parsed.ticker}")
        print(f"Fiscal Year:            {parsed.fiscal_year}")
        print(f"Target Quarters:        {parsed.target_quarters}")
        print(f"Is Comparative:         {parsed.is_comparative}")
        print(f"Core Topics:            {parsed.core_topics}")
        print(f"Optimized Search Query: {parsed.optimized_search_query}")

if __name__ == "__main__":
    main()