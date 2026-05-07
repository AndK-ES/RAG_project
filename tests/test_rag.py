#!/usr/bin/env python3
"""
Test script for the RAG pipeline
"""

import os
from src.rag_pipeline import RAGPipeline
from src import config

def test_rag_pipeline():
    """Test the RAG pipeline"""
    print("Starting RAG pipeline testing")
    print("=" * 50)

    try:
        if not config.GROQ_API_KEY:
            print("Error: GROQ_API_KEY not found in environment variables")
            print("Create a .env file with: GROQ_API_KEY=your_key_here")
            return

        print("API key found")

        print("Initializing RAG pipeline...")
        rag = RAGPipeline()
        print("RAG pipeline initialized")

        test_questions = [
            "Что такое живопись?",
            "Расскажи о скульптуре",
            "Какие бывают виды искусства?",
            "Что такое музейные экспонаты?"
        ]

        print("\n" + "=" * 50)
        print("TESTING QUESTIONS")
        print("=" * 50)

        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 40)

            result = rag.ask_question(question, top_k=3)

            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing time: {result['total_processing_time']:.2f}s")
            print(f"Sources found: {result['total_sources']}")

            if result['sources']:
                print("\nSources:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"  {j}. {source['source_file']} (relevance: {source['relevance_score']:.4f})")
                    print(f"     {source['content'][:100]}...")

            print("\n" + "=" * 50)

        print("\nTesting completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        print("Check logs in rag_system.log")

def interactive_mode():
    """Interactive mode for testing"""
    print("INTERACTIVE TESTING MODE")
    print("Enter 'quit' to exit")
    print("=" * 50)

    try:
        rag = RAGPipeline()

        while True:
            question = input("\nEnter your question: ").strip()

            if question.lower() == 'quit':
                print("Goodbye!")
                break

            if not question:
                print("Please enter a question")
                continue

            print(f"\nProcessing question: '{question}'")
            print("-" * 40)

            result = rag.ask_question(question, top_k=5)

            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing time: {result['total_processing_time']:.2f}s")

            if result['sources']:
                print(f"\nSources ({result['total_sources']}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['source_file']} (relevance: {source['relevance_score']:.4f})")

            print("-" * 40)

    except Exception as e:
        print(f"Error in interactive mode: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        test_rag_pipeline()
