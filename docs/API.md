# Museum Art RAG API

A REST API and web interface for answering questions about museum art and exhibits using RAG (Retrieval-Augmented Generation).

## Features

- **REST API** with comprehensive documentation
- **Web Interface** with modern, responsive design
- **Input Validation** using Pydantic models
- **Swagger/OpenAPI Documentation** automatically generated
- **Health Check Endpoint** for monitoring
- **Search Endpoint** for exploring the knowledge base

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Make sure you have the following environment variables set in your `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama3-8b-8192
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048
CHROMA_DB_DIR=./chroma_db
DEFAULT_TOP_K=5
```

### 3. Run the Server

```bash
python run_server.py
```

Or alternatively:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation (Swagger)**: http://localhost:8000/docs
- **Alternative API Documentation (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### 1. Health Check

**GET** `/health`

Check the health status of the API and RAG pipeline.

**Response:**
```json
{
  "status": "healthy",
  "rag_pipeline_ready": true
}
```

### 2. Ask Question

**POST** `/api/ask`

Ask a question about museum art and exhibits.

**Request Body:**
```json
{
  "question": "Tell me about Vincent van Gogh's paintings",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Vincent van Gogh was a Dutch post-impressionist painter...",
  "sources": [
    {
      "content": "Vincent van Gogh (1853-1890) was a Dutch post-impressionist painter...",
      "source_file": "objects/100/100001.json",
      "chunk_id": "chunk_1",
      "relevance_score": 0.9234
    }
  ],
  "confidence": 0.9234,
  "total_sources": 1,
  "generation_time": 2.45
}
```

### 3. Search Documents

**GET** `/api/search?query=van gogh&top_k=5`

Search for relevant documents without generating an answer.

**Response:**
```json
{
  "query": "van gogh",
  "results": [
    {
      "content": "Document content...",
      "metadata": {
        "source_file": "objects/100/100001.json",
        "chunk_id": "chunk_1"
      },
      "relevance_score": 0.9234
    }
  ],
  "total_found": 1
}
```

## Web Interface

The web interface provides a user-friendly way to interact with the RAG system:

### Features:
- **Modern Design**: Clean, responsive interface with gradient backgrounds
- **Real-time Processing**: Loading indicators and progress feedback
- **Source Information**: Display of information sources with relevance scores
- **Confidence Indicators**: Visual confidence badges (High/Medium/Low)
- **Statistics**: Generation time and source count
- **Mobile Responsive**: Works on desktop and mobile devices

### Usage:
1. Enter your question in the text area
2. Select the number of sources to retrieve (3-10)
3. Click "Ask Question"
4. View the generated answer and source information

## API Documentation

### Swagger UI
Visit http://localhost:8000/docs for interactive API documentation where you can:
- Test endpoints directly
- View request/response schemas
- See example requests and responses

### ReDoc
Visit http://localhost:8000/redoc for an alternative documentation view.

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input parameters
- **503 Service Unavailable**: RAG pipeline not ready
- **500 Internal Server Error**: Server-side errors

All errors include descriptive messages to help with debugging.

## Development

### Project Structure
```
project_new/
├── api.py                 # FastAPI application
├── run_server.py          # Server startup script
├── rag_pipeline.py        # RAG pipeline implementation
├── config.py              # Configuration settings
├── templates/
│   └── index.html         # Web interface template
├── requirements.txt       # Python dependencies
└── API_README.md          # This file
```

### Adding New Endpoints

To add new endpoints, modify `api.py`:

```python
@app.get("/api/new-endpoint")
async def new_endpoint():
    return {"message": "New endpoint"}
```

### Customizing the Web Interface

The web interface is in `templates/index.html` and includes:
- Modern CSS with gradients and animations
- JavaScript for form handling and API calls
- Responsive design for mobile devices

## Troubleshooting

### Common Issues

1. **RAG Pipeline Not Ready**
   - Check that your environment variables are set correctly
   - Ensure the ChromaDB directory exists and contains data
   - Check the logs for initialization errors

2. **API Key Issues**
   - Verify your GROQ_API_KEY is valid
   - Check that the API key has sufficient credits

3. **Port Already in Use**
   - Change the port in `run_server.py` or use a different port
   - Kill existing processes using the port

### Logs

The application logs important events including:
- RAG pipeline initialization
- Question processing
- Error messages
- Performance metrics

Check the console output for detailed logs.

## Performance

- **Typical Response Time**: 2-5 seconds for most questions
- **Concurrent Requests**: The API can handle multiple concurrent requests
- **Memory Usage**: Depends on the size of the knowledge base

## Security

- Input validation prevents malicious requests
- No sensitive data is exposed in responses
- API keys are kept secure in environment variables

## License

This project is part of the Museum Art RAG System. 