# src/router.py
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class QueryIntent(BaseModel):
    ticker: Optional[str] = Field(
        default=None, 
        description="Company stock ticker symbol if mentioned or implied (e.g., 'MSFT', 'NVDA')."
    )
    fiscal_year: Optional[int] = Field(
        default=None, 
        description="Fiscal year if specified (e.g., 2025, 2026)."
    )
    target_quarters: List[str] = Field(
        default_factory=list, 
        description="List of fiscal quarters explicitly targeted for comparison (e.g., ['Q1', 'Q2']). Must use uppercase like 'Q1', 'Q2', 'Q3', 'Q4'."
    )
    is_comparative: bool = Field(
        description="True if the query compares two or more quarters or seeks guidance evolution; False if asking about a single quarter or general topic."
    )
    core_topics: List[str] = Field(
        description="Key financial metrics or subjects targeted (e.g., ['CapEx', 'Azure', 'operating margin', 'useful life'])."
    )
    optimized_search_query: str = Field(
        description="A concise, dense retrieval query stripped of pleasantries and conversational filler."
    )

ROUTER_SYSTEM_PROMPT = """You are a financial query parser for an earnings call RAG system.
Your job is to inspect user prompts and extract structured routing parameters.

Rules:
- Identify target quarters (Q1, Q2, Q3, Q4). If the user mentions 'first quarter and second quarter', map to ['Q1', 'Q2'].
- If the user asks about a single period, set target_quarters to that single quarter and is_comparative to False.
- If no quarters are mentioned, leave target_quarters empty.
- Generate an optimized_search_query that extracts the core factual/conceptual question for vector and BM25 search.
"""

class QueryRouter:
    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            max_retries=3
        )
        self.structured_llm = llm.with_structured_output(QueryIntent)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("user", "User Query: {query}")
        ])
        self.chain = self.prompt | self.structured_llm

    def route(self, query: str) -> QueryIntent:
        """Parses a raw string query into structured QueryIntent."""
        return self.chain.invoke({"query": query})