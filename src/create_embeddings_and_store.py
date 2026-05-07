from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import json

CHROMA_DB_DIR = 'chroma_db'
CHUNKS_FILE = 'data_chunks.json'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def search_relevant_documents(query, top_k=5, collection_name=None):
    """
    Search for relevant documents by query

    Args:
        query (str): Search query
        top_k (int): Number of results to return
        collection_name (str, optional): Collection name (default if None)

    Returns:
        list: List of relevant documents with metadata
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        if collection_name:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=collection_name
            )
        else:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )

        results = vectorstore.similarity_search_with_relevance_scores(
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

        return formatted_results

    except Exception as e:
        print(f"Error during search: {e}")
        return []

def demonstrate_search():
    """
    Demonstrate the search function usage
    """
    print("\n" + "="*50)
    print("SEARCH DEMONSTRATION")
    print("="*50)

    test_queries = [
        "живопись",
        "скульптура",
        "история искусства",
        "музейные экспонаты",
        "художники"
    ]

    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        print("-" * 40)

        results = search_relevant_documents(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Relevance: {result['relevance_score']:.4f}")
                print(f"   Source: {result['metadata'].get('source_file', 'Unknown')}")
                print(f"   Chunk ID: {result['metadata'].get('chunk_id', 'Unknown')}")
                print(f"   Content: {result['content'][:200]}...")
        else:
            print("No results found")

print("Loading chunks...")
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

chunks = chunks[:5000]
print(f"Processing {len(chunks)} chunks (out of {len(json.load(open(CHUNKS_FILE, 'r', encoding='utf-8')))} available)...")

texts = [chunk['text'] for chunk in chunks]
metadatas = [{
    'source_file': chunk['source_file'],
    'chunk_id': chunk['chunk_id']
} for chunk in chunks]

print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Creating embeddings and saving to ChromaDB...")
print(f"Generating embeddings for {len(texts)} chunks...")
print("This may take a few minutes...")

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=CHROMA_DB_DIR
)

print(f"Saved {len(texts)} embeddings to ChromaDB ({CHROMA_DB_DIR})")

if __name__ == "__main__":
    demonstrate_search()
