# Earnings Call Guidance & Delta Tracker (Domain-Specific RAG)

A specialized Retrieval-Augmented Generation (RAG) system built to track financial guidance shifts, CapEx allocation, and executive tone across quarterly earnings calls (Q1 vs. Q2).

Unlike naive RAG pipelines that arbitrarily slice text by token counts, this architecture enforces:
1. **Deterministic Structural Parsing:** Separates prepared management remarks from interactive Q&A sessions.
2. **Atomic Q&A Binding:** Keeps analyst inquiries bound directly to executive responses to preserve conversational context.
3. **Hybrid Sparse + Dense Indexing:** Combines dense semantic vector embeddings (`BAAI/bge-small-en-v1.5`) with sparse keyword matching (`BM25`) via Reciprocal Rank Fusion (RRF).
4. **Temporal Partitioned Retrieval:** Guarantees balanced retrieval across target quarters to prevent recent quarters from drowning out past baselines.
5. **Cross-Encoder Reranking:** Re-scores candidates per quarter with FlashRank (`ms-marco-TinyBERT-L-2-v2`).
6. **Forensic Financial Synthesis:** Instructs the LLM to compute concrete period-over-period deltas with strict citations.

---

## Architecture Flow

```
[Raw Transcripts (TXT/DOCX)]
           │
           ▼
[Parser: Regex Turn Segmentation]
   ├── Prepared Remarks (Split via RecursiveCharacterTextSplitter)
   └── Q&A Exchanges (Paired Atomically: Analyst -> Executive)
           │
           ▼
[Hybrid Retrieval Engine]
   ├── Dense Vector Index (ChromaDB + bge-small-en-v1.5)
   └── Sparse Keyword Index (BM25)
           │
           ▼
[Reciprocal Rank Fusion (RRF)]
           │
           ▼
[Temporal Partitioning & Cross-Encoder Rerank]
   ├── Quarter 1 Slice -> Top Reranked Passages
   └── Quarter 2 Slice -> Top Reranked Passages
           │
           ▼
[Grounded Synthesis LLM (Gemini)]
           │
           ▼
[Executive Delta Analysis & Citations Table]
```

---

## Getting Started

### 1. Clone & Set Up Environment

```bash
git clone [https://github.com/](https://github.com/)<your-username>/earnings-call-rag.git
cd earnings-call-rag

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the template environment file:

```bash
cp .env.example .env
```

Open `.env` and provide your Google AI Studio API key:

```env
GOOGLE_API_KEY="your_actual_key_here"
```

### 3. Add Raw Transcripts

Place your transcript text files in `data/raw/` following this naming convention:
* `data/raw/MSFT_FY26_Q1.txt`
* `data/raw/MSFT_FY26_Q2.txt`

### 4. Run the Pipeline

```bash
python run_pipeline.py
```

---

## How to Customize Models and LLM Providers

All LLM initialization logic lives inside `src/synthesizer.py`.

### 1. Switching Between Gemini Models
To use different Gemini model checkpoints (e.g., higher reasoning capability vs. higher throughput), update the `model_name` parameter in `src/synthesizer.py`:

```python
# In src/synthesizer.py:
synthesizer = EarningsSynthesizer(model_name="gemini-3.5-flash-lite", temperature=0.0)

# Options:
# - "gemini-3.5-flash-lite" (Ultra-fast, high rate limits)
# - "gemini-3.6-flash" / "gemini-flash-latest" (Balanced)
# - "gemini-2.5-pro" (Deep analytical reasoning)
```

### 2. Switching to a Local LLM (Ollama)
If you want to run completely offline with no API keys:

1. Install Ollama and pull your model:
   ```bash
   ollama pull llama3.1:8b
   ```
2. In `src/synthesizer.py`, replace the `ChatGoogleGenerativeAI` instance:
   ```python
   from langchain_community.chat_models import ChatOllama

   class EarningsSynthesizer:
       def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.0):
           self.llm = ChatOllama(model=model_name, temperature=temperature)
           # The prompt and chain remain unchanged
   ```

### 3. Switching to Groq / OpenAI
Install the provider integration:
```bash
pip install langchain-groq  # or langchain-openai
```

In `src/synthesizer.py`:
```python
# For Groq:
from langchain_groq import ChatGroq
self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

# For OpenAI:
from langchain_openai import ChatOpenAI
self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
```