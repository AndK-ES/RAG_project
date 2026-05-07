# Museum Art RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions about museum art and exhibits. Uses **Groq API** (llama3-70b-8192) for LLM inference, **sentence-transformers** for embeddings, and **ChromaDB** as the vector database. Includes a **FastAPI** web server with a modern web interface and Swagger-documented REST API.

The art collection data comes from the **Minneapolis Institute of Art (Mia)** via the [`artsmia/collection`](https://github.com/artsmia/collection) dataset.

## Project Structure

```
project/
├── src/                          # Source code
│   ├── config.py                 # Configuration (API keys, model params)
│   ├── rag_data_preparation.py   # Data extraction and chunking
│   ├── create_embeddings_and_store.py  # Embedding generation and storage
│   ├── rag_pipeline.py           # Core RAG pipeline (search + generation)
│   ├── api.py                    # FastAPI REST API
│   └── run_server.py             # Server launcher
├── tests/                        # Test scripts
│   ├── test_rag.py               # RAG pipeline tests
│   └── test_api.py               # API endpoint tests
├── docs/                         # Documentation
│   └── API.md                    # API reference
├── templates/
│   └── index.html                # Web interface
├── collection-main/              # Mia art collection data (submodule)
├── requirements.txt              # Python dependencies
├── start_server.bat              # Windows server launcher
└── README.md                     # This file
```

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get an API key from [Groq Console](https://console.groq.com/).

### 3. Prepare data (if not already done)

Extract and chunk the art collection data:

```bash
python -m src.rag_data_preparation
```

### 4. Generate embeddings (if not already done)

Create embeddings and store them in ChromaDB:

```bash
python -m src.create_embeddings_and_store
```

## Usage

### Start the API server

```bash
python -m src.run_server
```

Then open:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Test the RAG pipeline

```bash
python -m tests.test_rag
```

### Interactive mode

```bash
python -m tests.test_rag --interactive
```

### Use in code

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

result = rag.ask_question("Tell me about Vincent van Gogh", top_k=5)

print(result['answer'])
print(f"Confidence: {result['confidence']}")
print(f"Sources: {result['sources']}")
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/api/ask` | Ask a question (RAG) |
| GET | `/api/search` | Search documents only |

See [docs/API.md](docs/API.md) for full API reference.

## Configuration

Key parameters in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `llama3-70b-8192` | LLM model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DEFAULT_TOP_K` | `5` | Default number of documents to retrieve |
| `LLM_TEMPERATURE` | `0.1` | LLM generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Maximum output tokens |

## Features

- **RAG Pipeline**: Combines vector search with LLM generation for grounded answers
- **Dual Interface**: Web UI + REST API with Swagger/ReDoc documentation
- **Source Transparency**: Every answer includes source documents with relevance scores
- **Confidence Scoring**: Average relevance score across retrieved documents
- **Comprehensive Logging**: All operations logged to `rag_system.log` and console

## Requirements

- Python 3.8+
- Groq API key
- Internet connection for embedding model download

## Troubleshooting

1. **API key error**: Ensure `.env` file exists with a valid `GROQ_API_KEY`
2. **Model download error**: Check your internet connection
3. **ChromaDB error**: Ensure embeddings have been generated (`python -m src.create_embeddings_and_store`)
4. **Timeouts**: Increase `LLM_TIMEOUT` in configuration

## License

The art collection data is provided under [CC0 1.0 Universal](https://github.com/artsmia/collection/blob/main/LICENSE) (Public Domain Dedication) by the Minneapolis Institute of Art.
