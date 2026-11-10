# src/parser.py
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def parse_transcript_turns(raw_text: str) -> List[Dict[str, Any]]:
    """
    Splits raw transcript text into structured individual speech turns.
    Identifies speaker, affiliation, and section boundary (Prepared vs Q&A).
    """
    lines = raw_text.splitlines()
    in_qa_section = False
    
    # Matches patterns like:
    # "SATYA NADELLA:" or "AMY HOOD:" or "MARK MOERDLER, Bernstein:"
    speaker_pattern = re.compile(r"^([A-Z\s\.\-]+)(?:,\s*([A-Za-z0-9\s\.\&\-]+))?:\s*(.*)$")
    
    turns: List[Dict[str, Any]] = []
    current_speaker = None
    current_affiliation = None
    current_buffer = []

    def flush_turn():
        nonlocal current_speaker, current_affiliation, current_buffer
        if current_speaker and current_buffer:
            text = " ".join(current_buffer).strip()
            # Drop operator handoffs and empty lines
            if current_speaker.upper() != "OPERATOR" and text:
                role = "Executive"
                if current_affiliation:
                    role = "Analyst"
                elif "INVESTOR RELATIONS" in current_speaker.upper():
                    role = "IR"
                
                turns.append({
                    "speaker": current_speaker.title(),
                    "affiliation": current_affiliation if current_affiliation else "Microsoft",
                    "role": role,
                    "section": "qa" if in_qa_section else "prepared_remarks",
                    "text": text
                })
        current_buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect boundary into Q&A
        if "(Operator Direction.)" in line or "Operator, next question, please." in line or "Question-and-Answer Session" in line:
            in_qa_section = True
            continue

        match = speaker_pattern.match(line)
        if match:
            flush_turn()
            current_speaker = match.group(1).strip()
            current_affiliation = match.group(2).strip() if match.group(2) else None
            initial_text = match.group(3).strip()
            if initial_text:
                current_buffer.append(initial_text)
        else:
            if current_speaker:
                current_buffer.append(line)
                
    flush_turn()
    return turns


def turns_to_langchain_documents(
    turns: List[Dict[str, Any]], 
    ticker: str, 
    fiscal_year: int, 
    fiscal_quarter: str
) -> List[Document]:
    """
    Converts speech turns into optimized retrieval chunks:
    - Prepared remarks: Split with RecursiveCharacterTextSplitter
    - Q&A: Paired atomically (Analyst Question + Executive Response)
    """
    docs: List[Document] = []
    
    # Text splitter reserved only for long prepared remarks
    remarks_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    i = 0
    while i < len(turns):
        turn = turns[i]
        
        # 1. Prepared Remarks
        if turn["section"] == "prepared_remarks":
            raw_content = f"{turn['speaker']} ({turn['role']}): {turn['text']}"
            sub_chunks = remarks_splitter.split_text(raw_content)
            
            for chunk_idx, chunk_text in enumerate(sub_chunks):
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": fiscal_quarter,
                        "section": "prepared_remarks",
                        "speaker": turn["speaker"],
                        "role": turn["role"],
                        "chunk_index": chunk_idx
                    }
                )
                docs.append(doc)
            i += 1
            
        # 2. Q&A Session
        elif turn["section"] == "qa":
            # If current turn is an analyst question followed by an executive response
            if turn["role"] == "Analyst" and i + 1 < len(turns) and turns[i + 1]["role"] == "Executive":
                q_turn = turn
                a_turn = turns[i + 1]
                
                content = (
                    f"Question from {q_turn['speaker']} ({q_turn['affiliation']}):\n"
                    f"{q_turn['text']}\n\n"
                    f"Response by {a_turn['speaker']} ({a_turn['role']}):\n"
                    f"{a_turn['text']}"
                )
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": fiscal_quarter,
                        "section": "qa",
                        "analyst_name": q_turn["speaker"],
                        "analyst_firm": q_turn["affiliation"],
                        "executive_name": a_turn["speaker"],
                        "executive_role": a_turn["role"]
                    }
                )
                docs.append(doc)
                i += 2  # Consumed both question and answer
            else:
                # Standalone executive remarks or follow-up clarifications during Q&A
                doc = Document(
                    page_content=f"{turn['speaker']} ({turn['role']}): {turn['text']}",
                    metadata={
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": fiscal_quarter,
                        "section": "qa",
                        "speaker": turn["speaker"],
                        "role": turn["role"]
                    }
                )
                docs.append(doc)
                i += 1
        else:
            i += 1
            
    return docs


def load_transcript(file_path: str, ticker: str, fiscal_year: int, fiscal_quarter: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    turns = parse_transcript_turns(raw_text)
    return turns_to_langchain_documents(turns, ticker, fiscal_year, fiscal_quarter)