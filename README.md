# Jerusalem RAG Explorer

A Retrieval-Augmented Generation (RAG) application for researching the **Kingdom of Jerusalem** and the **Crusades** (1095-1291 CE). Ask questions about Crusader history and receive AI-generated answers grounded in primary source documents.

## Features

- **Multilingual Corpus**: Latin, Arabic, Greek, Armenian, and French medieval manuscripts
- **Pre-translation Pipeline**: Non-English sources are translated during ingestion for better retrieval
- **Multiple Response Modes**: Scholarly answers, timelines, dossiers, fact-checking, source comparison
- **Source Attribution**: Every answer cites specific source chunks with language and author metadata

## Versions

| Version | File | Description |
|---------|------|-------------|
| **v1** | `app.py` | English-only corpus from Archive.org and Wikipedia |
| **v2** | `app_v2.py` | Multilingual corpus with translation support |

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key (free tier works)

### Installation

```bash
# Clone and enter directory
git clone https://github.com/yourusername/jerusalem_rag.git
cd jerusalem_rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo GEMINI_API_KEY=your_key_here > .env
```

### Running v1 (English Only)

```bash
# Fetch English documents
python -m rag.archive_fetch
python -m rag.wiki_fetch

# Build index
python -m rag.injest

# Run app
streamlit run app.py
```

### Running v2 (Multilingual)

```bash
# Fetch multilingual manuscripts from Archive.org
python -m rag.archive_fetch_v2

# Build index with translation (may take time due to API rate limits)
python -m rag.injest_v2

# Run app
streamlit run app_v2.py
```

## Project Structure

```
jerusalem_rag/
├── app.py                  # v1 Streamlit interface
├── app_v2.py               # v2 Multilingual interface
├── requirements.txt        # Python dependencies
├── .env                    # GEMINI_API_KEY (create this)
│
├── rag/
│   ├── gemini_client.py    # Gemini API wrapper
│   │
│   ├── # v1 modules
│   ├── archive_fetch.py    # Fetch English texts from Archive.org
│   ├── wiki_fetch.py       # Fetch Wikipedia articles
│   ├── injest.py           # v1 chunking and indexing
│   ├── retrieve.py         # v1 retrieval
│   ├── prompts.py          # v1 prompt templates
│   │
│   ├── # v2 modules
│   ├── models.py           # Data models with language metadata
│   ├── archive_fetch_v2.py # Fetch multilingual manuscripts
│   ├── gallica_fetch.py    # Fetch from BnF Gallica
│   ├── translation.py      # Translation pipeline with caching
│   ├── injest_v2.py        # Multilingual ingestion
│   ├── retrieve_v2.py      # Language-aware retrieval
│   └── prompts_v2.py       # Enhanced prompt templates
│
└── data/
    ├── raw/
    │   ├── archive/        # v1 English documents
    │   ├── archive_v2/     # v2 Multilingual manuscripts
    │   └── wiki/           # Wikipedia articles
    ├── translations/       # Cached translations
    ├── index/              # v1 FAISS index
    └── index_v2/           # v2 FAISS index
```

## How It Works

### RAG Pipeline

1. **Fetch**: Download historical texts from Archive.org, Wikipedia, or Gallica
2. **Chunk**: Split documents into 2000-character segments with 300-char overlap
3. **Translate** (v2): Translate non-English chunks to English using Gemini API
4. **Embed**: Generate embeddings using `all-MiniLM-L6-v2` (384 dimensions)
5. **Index**: Store embeddings in FAISS for fast similarity search
6. **Retrieve**: Find top-k relevant chunks for user queries
7. **Generate**: Send context + question to Gemini LLM for answer generation

### v2 Source Languages

| Language | Sources |
|----------|---------|
| Latin | Recueil des historiens - Historiens occidentaux (William of Tyre, Fulcher of Chartres) |
| Arabic | Recueil des historiens - Historiens orientaux (Ibn al-Athir, Usama ibn Munqidh) |
| Greek | Recueil des historiens - Historiens grecs (Anna Comnena's Alexiad) |
| Armenian | Recueil des historiens - Documents armeniens |
| French | Assises de Jerusalem (Laws of the Kingdom), Hayton's chronicles |

## Response Modes

| Mode | Description |
|------|-------------|
| **Default** | Scholarly answer with inline citations |
| **Chronology** | Timeline format with dates and events |
| **Dossier** | Structured report with sections |
| **Claim Check** | Fact-check historical claims with confidence rating |
| **Comparative** (v2) | Compare Latin, Arabic, and Greek source perspectives |
| **Retrieval** | List relevant passages without synthesis |

## Configuration

### Ingestion Options (v2)

```bash
python -m rag.injest_v2 --help

Options:
  --skip-translation    Skip translation (faster, but retrieval quality drops)
  --max-translate N     Limit translations (default: 15, Gemini free tier safe)
  --data-dir PATH       Source document directory
  --index-dir PATH      Output index directory
```

### Rate Limits

The free Gemini API tier has limits:
- **5 requests/minute** (RPM)
- **20 requests/day** (RPD) for some models

The translation pipeline uses 4 RPM to stay safely under limits.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Google Gemini (gemini-3-flash-preview) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (IndexFlatIP) |
| Translation | Gemini API |
| Language Detection | langdetect |

## Example Questions

- "What happened at the Battle of Hattin?"
- "Who was Baldwin IV of Jerusalem?"
- "What do Arabic sources say about Saladin?"
- "Compare Latin and Arabic accounts of the siege of Acre"
- "What were the laws of the Kingdom of Jerusalem?"

## License

MIT

## Acknowledgments

- **Archive.org** for digitized historical texts
- **Gallica (BnF)** for manuscript access
- **Wikipedia** for encyclopedic content
- Historical scholars who created the *Recueil des historiens des croisades*
