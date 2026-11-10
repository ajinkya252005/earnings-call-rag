# src/synthesizer.py
import os
from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = """You are a senior financial analyst and forensic accountant.
Your task is to analyze earnings call commentary across fiscal quarters and identify:
1. Exact quantitative figures, ranges, and commitments.
2. Strategic priorities and commentary from management.
3. Concrete shifts, revisions, or tone changes between periods (the "Delta").

STRICT GROUNDING RULES:
- Only make claims that are explicitly backed by the provided excerpts.
- Cite the speaker, title/firm, and quarter for every claim (e.g., [Amy Hood, CFO, Q1]).
- If a metric was not mentioned or guidance was unchanged, explicitly state that rather than assuming.
- Do not invent numbers or round off stated percentages.
"""

USER_TEMPLATE = """User Query: {query}

Here is the retrieved context chronologically organized by fiscal quarter:

{formatted_context}

Analyze the changes across these quarters based ONLY on the excerpts above. 
Structure your response as:
1. Executive Summary
2. Quarter-by-Quarter Breakdown (Key Metrics & Strategic Focus)
3. Delta / Evolution Analysis (What specifically shifted between quarters?)
4. Verifiable Excerpts & Citations
"""

class EarningsSynthesizer:
    def __init__(self, model_name: str = "gemini-3.5-flash", temperature: float = 0.0):
        # ChatGoogleGenerativeAI automatically reads the GOOGLE_API_KEY environment variable
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=5
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_TEMPLATE)
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def format_context(self, comparative_results: Dict[str, List[Document]]) -> str:
        """Formats quarter-partitioned documents into chronological blocks."""
        formatted_blocks = []
        
        for quarter, docs in comparative_results.items():
            quarter_block = [f"=== FISCAL QUARTER: {quarter} ==="]
            for idx, doc in enumerate(docs, start=1):
                meta = doc.metadata
                speaker = meta.get("speaker") or f"{meta.get('analyst_name')} -> {meta.get('executive_name')}"
                role = meta.get("role") or meta.get("executive_role", "Executive")
                section = meta.get("section", "Transcript").upper()
                
                header = f"[Excerpt #{idx} | {section} | {speaker} ({role})]"
                content = doc.page_content.strip()
                quarter_block.append(f"{header}\n{content}\n")
            
            formatted_blocks.append("\n".join(quarter_block))
            
        return "\n\n" + ("\n" + "-" * 40 + "\n").join(formatted_blocks)

    def analyze(self, query: str, comparative_results: Dict[str, List[Document]]) -> str:
        formatted_context = self.format_context(comparative_results)
        return self.chain.invoke({
            "query": query,
            "formatted_context": formatted_context
        })