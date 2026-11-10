# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from src.parser import load_transcript
from src.indexer import build_hybrid_retriever
from src.engine import EarningsQueryEngine
from src.router import QueryRouter
from src.synthesizer import EarningsSynthesizer

load_dotenv()

st.set_page_config(
    page_title="Earnings Call Guidance & Delta Tracker",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Earnings Call Guidance & Delta Tracker")
st.caption("Forensic period-over-period financial guidance & capacity allocation analysis via Domain-Specific RAG.")

# Verify API key
if not os.environ.get("GOOGLE_API_KEY"):
    st.error("`GOOGLE_API_KEY` is not set. Please set it in your `.env` file.")
    st.stop()

@st.cache_resource(show_spinner=False)
def initialize_system():
    """Caches document loading, vector storage, and reranker initialization."""
    files = [
        ("data/raw/MSFT_FY26_Q1.txt", "MSFT", 2026, "Q1"),
        ("data/raw/MSFT_FY26_Q2.txt", "MSFT", 2026, "Q2"),
    ]
    
    all_docs = []
    for path, ticker, year, quarter in files:
        if os.path.exists(path):
            docs = load_transcript(path, ticker, year, quarter)
            all_docs.extend(docs)
    
    if not all_docs:
        return None, None, None, None

    hybrid_retriever = build_hybrid_retriever(all_docs, top_k=8)
    engine = EarningsQueryEngine(documents=all_docs, hybrid_retriever=hybrid_retriever)
    router = QueryRouter(model_name="gemini-3.5-flash-lite")
    synthesizer = EarningsSynthesizer(model_name="gemini-3.5-flash-lite")
    
    return all_docs, engine, router, synthesizer

with st.spinner("Initializing ChromaDB vector store and BM25 index..."):
    all_docs, engine, router, synthesizer = initialize_system()

if not all_docs:
    st.warning("No transcript files found in `data/raw/`. Please check file names and paths.")
    st.stop()

# Sidebar: System telemetry & manual controls
with st.sidebar:
    st.header("Indexed Corpus")
    st.metric("Total Ingested Chunks", len(all_docs))
    st.markdown("**Transcripts:**\n- `MSFT FY26 Q1`\n- `MSFT FY26 Q2`")
    st.divider()
    
    st.subheader("Query Configuration")
    top_k_per_q = st.slider("Chunks per quarter (Reranked)", min_value=1, max_value=5, value=2)
    st.caption("Temporal partitioning guarantees this exact count for each target quarter.")

# Query presets for easy testing
preset_queries = [
    "Compare Microsoft's CapEx guidance and GPU capacity allocation between Q1 and Q2.",
    "What did Amy Hood state regarding gross margin pressure from AI infrastructure scaling?",
    "How did Remaining Performance Obligations (RPO) and server useful life evolve across the two quarters?"
]

selected_preset = st.selectbox("Or choose a sample query:", [""] + preset_queries)
user_prompt = st.text_area(
    "Enter your financial inquiry:",
    value=selected_preset if selected_preset else "",
    height=80,
    placeholder="e.g., How did cloud capacity constraints impact Azure revenue growth between Q1 and Q2?"
)

if st.button("Run Financial Analysis", type="primary") and user_prompt:
    with st.status("Executing RAG Pipeline...", expanded=True) as status:
        # 1. Routing
        st.write("🔍 Parsing query intent and target quarters...")
        intent = router.route(user_prompt)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Quarters", ", ".join(intent.target_quarters) if intent.target_quarters else "All / Unspecified")
        with col2:
            st.metric("Comparative Mode", "Yes" if intent.is_comparative else "No")
        with col3:
            st.metric("Core Topics", ", ".join(intent.core_topics[:2]))

        # 2. Retrieval & Partitioning
        st.write("⚡ Executing hybrid search + cross-encoder reranking...")
        search_query = intent.optimized_search_query or user_prompt
        
        if len(intent.target_quarters) >= 2:
            comparative_results = engine.comparative_retrieve(
                search_query, 
                quarters=intent.target_quarters, 
                top_k_per_quarter=top_k_per_q
            )
        else:
            # Fallback to standard hybrid + rerank if 0 or 1 quarter detected
            active_quarter = intent.target_quarters[0] if intent.target_quarters else "Q2"
            raw_hits = engine.retriever.invoke(search_query, top_k=8)
            reranked_hits = engine.rerank_documents(search_query, raw_hits, top_n=top_k_per_q * 2)
            comparative_results = {active_quarter: reranked_hits}

        # 3. Synthesis
        st.write("🧠 Synthesizing comparative analysis via Gemini...")
        analysis = synthesizer.analyze(user_prompt, comparative_results)
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Render Synthesis
    st.subheader("Financial Synthesis & Delta Breakdown")
    st.markdown(analysis)

    # Collapsible Source Context
    st.divider()
    with st.expander("🔎 View Retrieved & Reranked Source Passages"):
        for quarter, docs in comparative_results.items():
            st.markdown(f"### Target Quarter: {quarter}")
            for idx, doc in enumerate(docs, start=1):
                meta = doc.metadata
                speaker = meta.get("speaker") or f"{meta.get('analyst_name')} ➔ {meta.get('executive_name')}"
                section = meta.get("section", "").upper()
                st.markdown(f"**[{quarter} | Hit #{idx} | {section} | {speaker}]**")
                st.info(doc.page_content)