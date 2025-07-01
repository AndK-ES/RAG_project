#!/usr/bin/env python3
"""
Simple script to run the Museum Art RAG API server
"""

import uvicorn
import logging

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Museum Art RAG API Server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🌐 Web Interface will be available at: http://localhost:8000")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run the server with import string format
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 