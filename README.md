# 📈 Earnings Call Guidance & Delta Tracker (Domain-Specific RAG)

A specialized, production-ready Retrieval-Augmented Generation (RAG) system built to track financial guidance shifts, capital expenditures (CapEx) allocation, and executive commentary across quarterly earnings calls (e.g., Q1 vs. Q2).

Standard RAG architectures fail on corporate earnings transcripts because naive character-count chunking cuts off speaker context, creates quarter bias, and hallucinates financial figures. This system solves those limitations with structural parsing, atomic dialogue binding, temporal query partitioning, and cross-encoder reranking.

---

## 🎯 Key Architectural Pillars

* **Deterministic Structural Ingestion:** Separates prepared executive presentations from spontaneous analyst Q&A blocks using heuristic turn detection.
* **Atomic Q&A Chunking:** Pairs analyst inquiries directly with executive responses into single retrieval units, ensuring answers retain the full context of the questions asked.
* **Hybrid Search (Dense + Sparse):** Integrates semantic dense embeddings (`BAAI/bge-small-en-v1.5`) with exact token matching (`BM25`) via custom Reciprocal Rank Fusion (RRF).
* **Temporal Partitioned Retrieval:** Prevents recency bias by fetching and scoring target quarters independently before merging candidates.
* **Cross-Encoder Reranking:** Applies FlashRank (`ms-marco-TinyBERT-L-2-v2`) locally to score passage relevance before context building.
* **Dynamic Query Routing:** Uses structured output parsing (Pydantic) to infer tickers, target quarters, and clean search terms from free-form queries.
* **Forensic Financial Synthesis:** Instructs the LLM to extract period-over-period delta tables with strict citations and zero ungrounded approximations.

---

## 🏗️ System Architecture

```text
[Raw Transcript (.txt / .docx)]
              │
              ▼
    [Structural Turn Parser]
       ├── Prepared Remarks ───► Recursive Splitter (Overlap: 150)
       └── Q&A Exchanges   ───► Atomic Pairing (Analyst ➔ Executive)
              │
              ▼
      [Dual Hybrid Indexing]
       ├── Dense Vector Store (ChromaDB + bge-small-en-v1.5)
       └── Sparse Inverted Index (BM25)
              │
              ▼
   [Dynamic Intent Router (Pydantic)] ──► Extracts Target Quarters & Clean Query
              │
              ▼
   [Temporal Partitioned Retrieval]
       ├── Quarter A Slice ──► Dense + BM25 (RRF) ──► FlashRank Rerank
       └── Quarter B Slice ──► Dense + BM25 (RRF) ──► FlashRank Rerank
              │
              ▼
  [Chronological Context Assembler]
              │
              ▼
      [Synthesizer LLM]
              │
              ▼
 [Grounded Delta Analysis + Source Citations]
```

---

## 📁 Project Structure

```text
earnings-call-rag/
├── data/
│   ├── raw/                  # Raw transcript text files (e.g., MSFT_FY26_Q1.txt)
│   ├── chroma_db/            # Persistent Chroma vector store (git-ignored)
│   └── models/               # Cached local reranker models (git-ignored)
├── src/
│   ├── __init__.py
│   ├── parser.py             # Domain-specific transcript cleaner & atomic chunker
│   ├── indexer.py            # ChromaDB + BM25 setup with custom RRF
│   ├── engine.py             # Partitioned retrieval & FlashRank cross-encoder reranker
│   ├── router.py             # Pydantic structured query decomposition
│   └── synthesizer.py        # LLM prompt orchestration & comparative formatting
├── app.py                    # Streamlit interactive web dashboard
├── run_pipeline.py           # End-to-end command-line runner
├── test_retrieval.py         # Ingestion & hybrid search sanity tester
├── test_router.py            # Dynamic intent decomposition tester
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/<your-username>/earnings-call-rag.git
cd earnings-call-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the template environment file:

```bash
cp .env.example .env
```

Open `.env` and add your API key:

```env
GOOGLE_API_KEY="your-google-ai-studio-api-key"
```

### 3. Add Transcript Files

Place your earnings transcripts in `data/raw/` using the naming pattern `<TICKER>_<YEAR>_<QUARTER>.txt`:

* `data/raw/MSFT_FY26_Q1.txt`
* `data/raw/MSFT_FY26_Q2.txt`

### 4. Run the Application

**Option A: Interactive Web Dashboard (Streamlit)**

```bash
streamlit run app.py
```

Open `http://localhost:8501` to use the interactive dashboard, view dynamic routing metrics, and inspect retrieved source passages.

**Option B: Command-Line Pipeline**

```bash
python run_pipeline.py
```

---

## 🔄 How to Switch or Customize LLM Providers

The LLM initialization logic is encapsulated within `src/synthesizer.py` and `src/router.py`. You can swap Google Gemini for local models (Ollama), Grok (xAI), OpenAI, Anthropic, or Groq by modifying the initialization class.

### LLM Provider Configuration Matrix

| Provider | Supported Model Examples | Required Package | Environment Variable | Implementation in `src/synthesizer.py` |
| --- | --- | --- | --- | --- |
| **Google Gemini** *(Default)* | `gemini-3.5-flash-lite`<br>`gemini-2.5-flash` | `langchain-google-genai` | `GOOGLE_API_KEY` | `from langchain_google_genai import ChatGoogleGenerativeAI`<br>`self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)` |
| **Grok (xAI)** | `grok-2-latest`<br>`grok-beta` | `langchain-openai` | `XAI_API_KEY` | `from langchain_openai import ChatOpenAI`<br>`self.llm = ChatOpenAI(model=model_name, api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1", temperature=0.0)` |
| **Local LLM (Ollama)** | `llama3.1:8b`<br>`mistral:7b` | `langchain-community` | *None (Local Server)* | `from langchain_community.chat_models import ChatOllama`<br>`self.llm = ChatOllama(model=model_name, temperature=0.0)` |
| **OpenAI** | `gpt-4o-mini`<br>`gpt-4o` | `langchain-openai` | `OPENAI_API_KEY` | `from langchain_openai import ChatOpenAI`<br>`self.llm = ChatOpenAI(model=model_name, temperature=0.0)` |
| **Anthropic** | `claude-3-5-sonnet-latest` | `langchain-anthropic` | `ANTHROPIC_API_KEY` | `from langchain_anthropic import ChatAnthropic`<br>`self.llm = ChatAnthropic(model=model_name, temperature=0.0)` |
| **Groq (Fast Inference)** | `llama-3.3-70b-versatile` | `langchain-groq` | `GROQ_API_KEY` | `from langchain_groq import ChatGroq`<br>`self.llm = ChatGroq(model_name=model_name, temperature=0.0)` |

---

### Step-by-Step Code Example: Switching Providers

#### Switching to Grok (xAI) or OpenAI

1. Install package:

   ```bash
   pip install langchain-openai
   ```

2. Add your key to `.env`:

   ```env
   # For Grok:
   XAI_API_KEY="your-xai-key"

   # For OpenAI:
   OPENAI_API_KEY="your-openai-key"
   ```

3. Update `src/synthesizer.py`:

   ```python
   import os
   from langchain_openai import ChatOpenAI

   class EarningsSynthesizer:
       def __init__(self, model_name: str = "grok-2-latest", temperature: float = 0.0):
           # Grok integration via xAI endpoint:
           self.llm = ChatOpenAI(
               model=model_name,
               api_key=os.getenv("XAI_API_KEY"),
               base_url="https://api.x.ai/v1",
               temperature=temperature
           )

           # OpenAI integration (gpt-4o / gpt-4o-mini):
           # self.llm = ChatOpenAI(model=model_name, temperature=temperature)
   ```

#### Switching to a Local Model via Ollama

1. Run your local Ollama instance:

   ```bash
   ollama run llama3.1:8b
   ```

2. Update `src/synthesizer.py`:

   ```python
   from langchain_community.chat_models import ChatOllama

   class EarningsSynthesizer:
       def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.0):
           self.llm = ChatOllama(model=model_name, temperature=temperature)
   ```

---

## 🧪 Testing Modules Independently

* **Test Parser & Turn Segmentation:**

  ```bash
  python src/parser.py
  ```

  Validates regular expression turn splitting, operator stripping, and Q&A pairing counts.

* **Test Hybrid Retrieval & RRF:**

  ```bash
  python test_retrieval.py
  ```

  Runs BM25 and Chroma vector searches and ranks hits via Reciprocal Rank Fusion.

* **Test Dynamic Intent Router:**

  ```bash
  python test_router.py
  ```

  Tests Pydantic structured query parsing across conversational prompts.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.