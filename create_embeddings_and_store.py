from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import json

# Параметры
CHROMA_DB_DIR = 'chroma_db'
CHUNKS_FILE = 'data_chunks.json'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def search_relevant_documents(query, top_k=5, collection_name=None):
    """
    Поиск релевантных документов по запросу
    
    Args:
        query (str): Поисковый запрос
        top_k (int): Количество возвращаемых результатов
        collection_name (str, optional): Имя коллекции (если None, используется дефолтная)
    
    Returns:
        list: Список релевантных документов с их метаданными
    """
    try:
        # Инициализация эмбеддингов
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Загрузка существующей базы данных
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
        
        # Поиск релевантных документов
        results = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=top_k
        )
        
        # Форматирование результатов
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': score
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        return []

def demonstrate_search():
    """
    Демонстрация использования функции поиска
    """
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ ПОИСКА РЕЛЕВАНТНЫХ ДОКУМЕНТОВ")
    print("="*50)
    
    # Примеры запросов для тестирования
    test_queries = [
        "живопись",
        "скульптура",
        "история искусства",
        "музейные экспонаты",
        "художники"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Поиск по запросу: '{query}'")
        print("-" * 40)
        
        results = search_relevant_documents(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Релевантность: {result['relevance_score']:.4f}")
                print(f"   Источник: {result['metadata'].get('source_file', 'Неизвестно')}")
                print(f"   ID чанка: {result['metadata'].get('chunk_id', 'Неизвестно')}")
                print(f"   Содержание: {result['content'][:200]}...")
        else:
            print("Результаты не найдены")

# Загрузка чанков
print("Загружаем чанки...")
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Ограничиваем для быстрого тестирования
chunks = chunks[:5000]  # Только первые 5000 чанков
print(f"Обрабатываем {len(chunks)} чанков (из {len(json.load(open(CHUNKS_FILE, 'r', encoding='utf-8')))} доступных)...")

# Тексты и метаданные
texts = [chunk['text'] for chunk in chunks]
metadatas = [{
    'source_file': chunk['source_file'],
    'chunk_id': chunk['chunk_id']
} for chunk in chunks]

print("Инициализируем модель эмбеддингов...")
# Инициализация эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Создаем эмбеддинги и сохраняем в ChromaDB...")
# Сохраняем в ChromaDB с отображением прогресса
print(f"Генерируем эмбеддинги для {len(texts)} чанков...")
print("Это может занять несколько минут...")

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=CHROMA_DB_DIR
)

print(f"✅ Сохранено {len(texts)} эмбеддингов в ChromaDB ({CHROMA_DB_DIR})")

# Демонстрация поиска после создания базы данных
if __name__ == "__main__":
    demonstrate_search()