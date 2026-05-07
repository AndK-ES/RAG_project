import logging
import time
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from src import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline"""
        logger.info("Initializing RAG pipeline...")

        try:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

            logger.info(f"Initializing LLM: {config.LLM_MODEL}")
            self.llm = ChatGroq(
                groq_api_key=config.GROQ_API_KEY,
                model_name=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )

            logger.info(f"Loading vector database: {config.CHROMA_DB_DIR}")
            self.vectorstore = Chroma(
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )

            self._create_prompt_template()

            logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise

    def _create_prompt_template(self):
        """Create the prompt template for answer generation"""
        system_prompt = """You are an expert in art and museum exhibits. Your task is to answer user questions using the provided information from the knowledge base.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided information
2. If the provided information does not contain the answer, honestly state that
3. Use information from sources to support your answers
4. Answer in the language of the user's question
5. Be accurate and informative
6. If information is contradictory, point it out
7. Structure your answer logically and clearly
8. Cite sources at the end

Provided information:
{context}

User question: {question}

Answer:"""

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        logger.info("Prompt template created")

    def search_documents(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Search for relevant documents

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant documents with metadata
        """
        start_time = time.time()
        logger.info(f"Searching documents for query: '{query}' (top_k={top_k})")

        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=top_k
            )

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': score
                })

            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s. Found {len(formatted_results)} documents")

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer based on found documents

        Args:
            question: User question
            context_docs: Found relevant documents

        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        logger.info(f"Generating answer for question: '{question}'")

        try:
            if not context_docs:
                logger.warning("No relevant documents for answer generation")
                return {
                    'answer': 'Sorry, I could not find relevant information to answer your question. Please try rephrasing your query.',
                    'sources': [],
                    'confidence': 0.0,
                    'generation_time': 0.0
                }

            context_parts = []
            for i, doc in enumerate(context_docs, 1):
                context_parts.append(
                    f"Source {i} (relevance: {doc['relevance_score']:.4f}):\n{doc['content']}"
                )
            context = "\n\n".join(context_parts)

            logger.info(f"Context formed from {len(context_docs)} documents")

            messages = self.prompt_template.format_messages(
                context=context,
                question=question
            )

            logger.info("Sending request to LLM...")
            response = self.llm.invoke(messages)
            answer = response.content

            sources = []
            for doc in context_docs:
                source_info = {
                    'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    'source_file': doc['metadata'].get('source_file', 'Unknown'),
                    'chunk_id': doc['metadata'].get('chunk_id', 'Unknown'),
                    'relevance_score': doc['relevance_score']
                }
                sources.append(source_info)

            avg_relevance = sum(doc['relevance_score'] for doc in context_docs) / len(context_docs)

            generation_time = time.time() - start_time
            logger.info(f"Answer generated in {generation_time:.2f}s. Confidence: {avg_relevance:.4f}")

            return {
                'answer': answer,
                'sources': sources,
                'confidence': avg_relevance,
                'total_sources': len(sources),
                'generation_time': generation_time
            }

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f'An error occurred while generating the answer: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'generation_time': generation_time
            }

    def ask_question(self, question: str, top_k: int = config.DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Full RAG process: search + answer generation

        Args:
            question: User question
            top_k: Number of documents to search

        Returns:
            Complete answer with sources and metadata
        """
        total_start_time = time.time()
        logger.info(f"Starting question processing: '{question}'")

        try:
            relevant_docs = self.search_documents(question, top_k)

            result = self.generate_answer(question, relevant_docs)

            result['search_query'] = question
            result['search_results_count'] = len(relevant_docs)
            result['total_processing_time'] = time.time() - total_start_time

            logger.info(f"Question processing completed in {result['total_processing_time']:.2f}s")

            return result

        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error(f"Error processing question: {e}")
            return {
                'answer': f'An error occurred while processing the question: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'search_query': question,
                'search_results_count': 0,
                'total_processing_time': total_time,
                'error': str(e)
            }

def main():
    """Demonstrate the RAG pipeline"""
    try:
        rag = RAGPipeline()

        test_questions = [
            "What is painting?",
            "Tell me about sculpture",
            "What kinds of art are there?",
            "What are museum exhibits?"
        ]

        print("=" * 60)
        print("RAG PIPELINE DEMONSTRATION")
        print("=" * 60)

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

            print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
