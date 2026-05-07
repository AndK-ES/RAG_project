from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import uvicorn
from src.rag_pipeline import RAGPipeline
from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Museum Art RAG API",
    description="A REST API for answering questions about museum art and exhibits using RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

rag_pipeline = None

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask about museum art and exhibits")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of relevant documents to retrieve (1-20)")

class SourceInfo(BaseModel):
    content: str = Field(..., description="Excerpt from the source document")
    source_file: str = Field(..., description="Source file name")
    chunk_id: Union[str, int] = Field(..., description="Chunk identifier")
    relevance_score: float = Field(..., description="Relevance score of the source")

class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[SourceInfo] = Field(..., description="List of information sources used")
    confidence: float = Field(..., description="Confidence score of the answer (0-1)")
    total_sources: int = Field(..., description="Total number of sources used")
    generation_time: float = Field(..., description="Time taken to generate the answer in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")
    rag_pipeline_ready: bool = Field(..., description="Whether the RAG pipeline is ready")

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    try:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        rag_pipeline = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check the health status of the API and RAG pipeline

    Returns:
        HealthResponse: Status information about the API
    """
    return HealthResponse(
        status="healthy" if rag_pipeline is not None else "unhealthy",
        rag_pipeline_ready=rag_pipeline is not None
    )

@app.post("/api/ask", response_model=QuestionResponse, tags=["Questions"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question about museum art and exhibits

    This endpoint uses RAG (Retrieval-Augmented Generation) to:
    1. Search for relevant documents in the knowledge base
    2. Generate an answer based on the found information
    3. Return the answer along with source information

    Args:
        request: QuestionRequest containing the question and optional parameters

    Returns:
        QuestionResponse: Generated answer with sources and metadata

    Raises:
        HTTPException: If RAG pipeline is not available or an error occurs
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not available. Please try again later."
        )

    try:
        logger.info(f"Processing question: {request.question}")

        result = rag_pipeline.ask_question(
            question=request.question,
            top_k=request.top_k or 5
        )

        sources = []
        for source in result['sources']:
            chunk_id = source.get('chunk_id', 'Unknown')
            if chunk_id is None:
                chunk_id = 'Unknown'
            elif isinstance(chunk_id, (int, float)):
                chunk_id = str(chunk_id)

            source_info = SourceInfo(
                content=source.get('content', ''),
                source_file=source.get('source_file', 'Unknown'),
                chunk_id=chunk_id,
                relevance_score=source.get('relevance_score', 0.0)
            )
            sources.append(source_info)

        response = QuestionResponse(
            answer=result.get('answer', 'No answer generated'),
            sources=sources,
            confidence=result.get('confidence', 0.0),
            total_sources=result.get('total_sources', len(sources)),
            generation_time=result.get('generation_time', 0.0)
        )

        logger.info(f"Successfully processed question. Confidence: {result.get('confidence', 0.0):.4f}")
        return response

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/search", tags=["Search"])
async def search_documents(query: str, top_k: int = 5):
    """
    Search for relevant documents without generating an answer

    This endpoint only performs document retrieval without answer generation.
    Useful for exploring the knowledge base or debugging.

    Args:
        query: Search query
        top_k: Number of documents to return (1-20)

    Returns:
        List of relevant documents with metadata
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not available. Please try again later."
        )

    if not query or len(query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20"
        )

    try:
        results = rag_pipeline.search_documents(query.strip(), top_k)
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

templates = Jinja2Templates(directory="templates")

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
