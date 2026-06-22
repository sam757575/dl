# FAQ RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) application demonstrating semantic search on FAQ data using vector embeddings and Chroma vector database.

## Architecture

```
┌─────────────────┐
│  FAQ Data       │  → XLSX with 25 Q&A pairs
│  (data/)        │     (25 FAQs across 5 categories)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │  → Reads & parses XLSX
│  (src/)         │     Normalizes text
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Embeddings (SentenceTransformers)
│  Model: all-MiniLM-L6-v2       │  → Converts text to vectors
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  → Chroma DB (persistent)
│  (Chroma)       │     Stores embeddings + metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Retriever      │  → Semantic search
│  (Cosine Sim)   │     Returns top-3 matches
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLI Interface  │  → Interactive user prompts
│  (User Loop)    │     Query → Retrieve → Display
└─────────────────┘
```

## Installation

1. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

2. **Verify structure:**
```
dlpractice/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── data/
│   └── faq_data.xlsx      # Sample FAQ data
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # XLSX loader
│   ├── vector_store.py    # Chroma DB setup
│   ├── retriever.py       # Semantic search
│   └── cli.py             # Interactive CLI
└── data/chroma_db/        # Vector DB (auto-created)
```

## Usage

### Running the Chatbot

```bash
python3 main.py
```

You'll see:
```
============================================================
     FAQ RAG Chatbot - Powered by Vector Database
============================================================

Commands:
  - Type your question to search FAQs
  - 'reset'  : Clear vector database and reload data
  - 'exit'   : Exit the application
============================================================
```

### Example Queries

```
>>> How do I reset my password?
============================================================
Match #1 (Relevance: 81.1%)
============================================================
Category: Accounts
Q: How do I reset my password?
A: To reset your password, go to the login page and click...
Tags: password, login, security
...

>>> What payment methods do you accept?
============================================================
Match #1 (Relevance: 76.6%)
============================================================
Category: Billing
Q: What payment methods do you accept?
A: We accept all major credit cards (Visa, Mastercard...
```

### Commands

| Command | Effect |
|---------|--------|
| `your question` | Search FAQ database |
| `reset` | Clear & reload vector store |
| `help` | Show available commands |
| `exit` | Quit application |

## Components

### 1. **Data Loader** (`src/data_loader.py`)
- Reads XLSX file with pandas
- Extracts Q&A pairs and metadata
- Combines question + answer for embedding

### 2. **Vector Store** (`src/vector_store.py`)
- Uses Chroma for persistent storage
- SentenceTransformers for embeddings
- Stores documents with metadata

### 3. **Retriever** (`src/retriever.py`)
- Encodes user queries
- Performs cosine similarity search
- Returns top-3 results with scores

### 4. **CLI** (`src/cli.py`)
- Interactive loop for user queries
- Pretty-formatted result display
- Database reset capability

## Key Features

✅ **Semantic Search** - Finds relevant FAQs by meaning, not keywords  
✅ **Persistent Storage** - Vector DB survives across sessions  
✅ **Fast Retrieval** - Cosine similarity on embeddings  
✅ **Lightweight Model** - all-MiniLM-L6-v2 (~22MB)  
✅ **Zero External Dependencies** - All local, no API keys needed  

## Performance

- **Model:** all-MiniLM-L6-v2 (22M parameters)
- **Embedding Time:** ~1 second for 25 docs
- **Query Time:** <100ms for semantic search
- **Similarity Score:** 0.0 to 1.0 (higher = more relevant)

## Sample Data

The app includes 25 sample FAQs across 5 categories:

- **Accounts** - Password reset, 2FA, team management
- **Billing** - Payments, refunds, upgrades, discounts
- **Products** - Shipping, features, reports, custom UI
- **Technical** - System requirements, file formats, API, backups
- **General** - Support, feedback, data security

Data stored in `data/faq_data.xlsx`

## Extending the System

### Add More FAQs
Edit `data/faq_data.xlsx` and run `reset` command in the app.

### Use Different Embeddings
In `src/vector_store.py`, change `model_name`:
```python
vector_store = VectorStore(model_name="all-mpnet-base-v2")  # More accurate
vector_store = VectorStore(model_name="all-MiniLM-L6-v2")   # Faster
```

### Integrate with LLM
Modify `src/retriever.py` to pass results to Claude/ChatGPT for answer generation.

## Troubleshooting

**Issue:** `ModuleNotFoundError`
```bash
pip3 install -r requirements.txt
```

**Issue:** Vector DB not persisting
- Check `./data/chroma_db/` directory exists
- Run `reset` command to force reload

**Issue:** Slow embedding on first run
- First run downloads SentenceTransformer model (~300MB)
- Subsequent runs use cached model

## Architecture Trade-offs

| Choice | Why |
|--------|-----|
| Chroma | Local, persistent, easy setup (vs FAISS) |
| all-MiniLM-L6-v2 | Balance of speed & accuracy |
| CLI | Simple demo (vs Streamlit for production) |
| No LLM | Focus on retrieval mechanics (can add Claude later) |
