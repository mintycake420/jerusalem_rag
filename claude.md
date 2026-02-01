# Jerusalem RAG Explorer

A **Retrieval-Augmented Generation (RAG)** application focused on the **Kingdom of Jerusalem and the Crusades**. Users can ask questions about Crusader history and receive AI-generated answers grounded in historical source documents.

## Versions

- **v1** (`app.py`): English-only corpus from Archive.org and Wikipedia
- **v2** (`app_v2.py`): Multilingual corpus with Latin, Arabic, and Greek manuscripts

## Architecture

```
jerusalem_rag/
├── app.py                 # v1 Streamlit web interface
├── app_v2.py              # v2 Multilingual UI
├── gemini.py              # API testing script
├── run.ps1                # Startup script
├── rag/
│   ├── # v1 modules
│   ├── archive_fetch.py   # Fetch texts from Archive.org
│   ├── wiki_fetch.py      # Fetch texts from Wikipedia
│   ├── injest.py          # v1 ingestion & indexing
│   ├── retrieve.py        # v1 retrieval
│   ├── gemini_client.py   # Gemini API wrapper
│   ├── prompts.py         # v1 prompt templates
│   │
│   ├── # v2 modules
│   ├── models.py          # Data models with language metadata
│   ├── archive_fetch_v2.py # Fetch multilingual texts from Archive.org
│   ├── translation.py     # Batch translation pipeline
│   ├── injest_v2.py       # Multilingual ingestion
│   ├── retrieve_v2.py     # Language-aware retrieval
│   └── prompts_v2.py      # Enhanced prompts
└── data/
    ├── raw/
    │   ├── archive/       # v1 Archive.org documents (English)
    │   ├── archive_v2/    # v2 Multilingual manuscripts
    │   └── wiki/          # Wikipedia articles
    ├── translations/      # Cached translations (v2)
    ├── index/             # v1 FAISS index
    └── index_v2/          # v2 multilingual index
```

## How It Works

### v1 Pipeline
1. Fetch English texts from Archive.org and Wikipedia
2. Chunk into 2000-char segments with 300-char overlap
3. Embed with `all-MiniLM-L6-v2` (384 dimensions)
4. Build FAISS index for similarity search
5. Query with Gemini LLM

### v2 Pipeline (Multilingual)
1. Fetch manuscripts from **Archive.org** (Recueil des historiens des croisades collection)
2. Detect language (English, Latin, Arabic, Greek, Armenian, French)
3. **Pre-translate** non-English chunks using Gemini API
4. Embed translated text for English query matching
5. Store both original and translation in metadata
6. Retrieve with language filtering
7. Display source language and original text

## Tech Stack

| Component    | Technology                               |
| ------------ | ---------------------------------------- |
| Frontend     | Streamlit                                |
| Embeddings   | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector DB    | FAISS                                    |
| LLM          | Google Gemini (free tier)                |
| Translation  | Gemini API (batch during ingestion)      |
| Lang Detect  | langdetect                               |
| Language     | Python 3                                 |

## v2 Source Languages

| Language | Sources |
|----------|---------|
| Latin    | Recueil des historiens des croisades - Historiens occidentaux (William of Tyre, etc.) |
| Arabic   | Recueil des historiens des croisades - Historiens orientaux (Ibn al-Athir, etc.) |
| Greek    | Recueil des historiens des croisades - Historiens grecs (Anna Comnena, etc.) |
| Armenian | Recueil des historiens des croisades - Documents arméniens |
| French   | Assises de Jérusalem (Laws of the Kingdom), Hayton's chronicles |
| English  | Archive.org chronicles, Wikipedia, scholarly translations |

## Running the App

### v1 (English only)
```powershell
streamlit run app.py
```

### v2 (Multilingual)
```powershell
# 1. Fetch multilingual manuscripts from Archive.org
python -m rag.archive_fetch_v2

# 2. Build multilingual index (includes translation)
python -m rag.injest_v2

# 3. Run the app
streamlit run app_v2.py
```

Requires a `GEMINI_API_KEY` in your `.env` file.

## v2 Response Modes

- **Default**: Scholarly answer with citations
- **Chronology**: Timeline with dates and events
- **Dossier**: Structured research dossier
- **Claim Check**: Fact-check historical claims
- **Comparative**: Compare Latin, Arabic, and Greek perspectives
- **Retrieval**: List relevant source passages

## Key Configuration

### v1
- Chunk size: 2000 characters
- Overlap: 300 characters
- Index: `data/index/`

### v2
- Chunk size: 2000 characters
- Overlap: 300 characters
- Index: `data/index_v2/`
- Translation cache: `data/translations/`
- Rate limit: 10 requests/minute (free tier safe)
