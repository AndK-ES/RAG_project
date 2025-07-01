import logging
import time
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        """Инициализация RAG пайплайна"""
        logger.info("Инициализация RAG пайплайна...")
        
        try:
            # Инициализация эмбеддингов
            logger.info(f"Загрузка модели эмбеддингов: {config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
            
            # Инициализация LLM
            logger.info(f"Инициализация LLM: {config.LLM_MODEL}")
            self.llm = ChatGroq(
                groq_api_key=config.GROQ_API_KEY,
                model_name=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )
            
            # Инициализация векторной базы данных
            logger.info(f"Загрузка векторной базы данных: {config.CHROMA_DB_DIR}")
            self.vectorstore = Chroma(
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )
            
            # Создание промпт-шаблона
            self._create_prompt_template()
            
            logger.info("RAG пайплайн успешно инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации RAG пайплайна: {e}")
            raise
    
    def _create_prompt_template(self):
        """Создание промпт-шаблона для генерации ответов"""
        system_prompt = """Ты - эксперт по искусству и музейным экспонатам. Твоя задача - отвечать на вопросы пользователей, используя предоставленную информацию из базы знаний.

ВАЖНЫЕ ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе предоставленной информации
2. Если в предоставленной информации нет ответа на вопрос, честно скажи об этом
3. Используй информацию из источников для подтверждения своих ответов
4. Отвечай на русском языке
5. Будь точным и информативным
6. Если информация противоречива, укажи на это
7. Структурируй ответ логично и понятно
8. В конце укажи источники информации

Предоставленная информация:
{context}

Вопрос пользователя: {question}

Ответ:"""
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        logger.info("Промпт-шаблон создан")
    
    def search_documents(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов
        
        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            
        Returns:
            Список релевантных документов с метаданными
        """
        start_time = time.time()
        logger.info(f"Поиск документов для запроса: '{query}' (top_k={top_k})")
        
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
            logger.info(f"Поиск завершен за {search_time:.2f}с. Найдено {len(formatted_results)} документов")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return []
    
    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Генерация ответа на основе найденных документов
        
        Args:
            question: Вопрос пользователя
            context_docs: Найденные релевантные документы
            
        Returns:
            Словарь с ответом и метаданными
        """
        start_time = time.time()
        logger.info(f"Генерация ответа для вопроса: '{question}'")
        
        try:
            if not context_docs:
                logger.warning("Нет релевантных документов для генерации ответа")
                return {
                    'answer': 'Извините, я не нашел релевантной информации для ответа на ваш вопрос. Попробуйте переформулировать запрос.',
                    'sources': [],
                    'confidence': 0.0,
                    'generation_time': 0.0
                }
            
            # Формируем контекст из найденных документов
            context_parts = []
            for i, doc in enumerate(context_docs, 1):
                context_parts.append(
                    f"Источник {i} (релевантность: {doc['relevance_score']:.4f}):\n{doc['content']}"
                )
            context = "\n\n".join(context_parts)
            
            logger.info(f"Контекст сформирован из {len(context_docs)} документов")
            
            # Генерируем ответ с помощью LLM
            messages = self.prompt_template.format_messages(
                context=context,
                question=question
            )
            
            logger.info("Отправка запроса к LLM...")
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Формируем список источников
            sources = []
            for doc in context_docs:
                source_info = {
                    'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    'source_file': doc['metadata'].get('source_file', 'Неизвестно'),
                    'chunk_id': doc['metadata'].get('chunk_id', 'Неизвестно'),
                    'relevance_score': doc['relevance_score']
                }
                sources.append(source_info)
            
            # Вычисляем общую уверенность на основе релевантности источников
            avg_relevance = sum(doc['relevance_score'] for doc in context_docs) / len(context_docs)
            
            generation_time = time.time() - start_time
            logger.info(f"Ответ сгенерирован за {generation_time:.2f}с. Уверенность: {avg_relevance:.4f}")
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': avg_relevance,
                'total_sources': len(sources),
                'generation_time': generation_time
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Ошибка при генерации ответа: {e}")
            return {
                'answer': f'Произошла ошибка при генерации ответа: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'generation_time': generation_time
            }
    
    def ask_question(self, question: str, top_k: int = config.DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Полный процесс RAG: поиск + генерация ответа
        
        Args:
            question: Вопрос пользователя
            top_k: Количество документов для поиска
            
        Returns:
            Полный ответ с источниками и метаданными
        """
        total_start_time = time.time()
        logger.info(f"Начало обработки вопроса: '{question}'")
        
        try:
            # Поиск релевантных документов
            relevant_docs = self.search_documents(question, top_k)
            
            # Генерация ответа
            result = self.generate_answer(question, relevant_docs)
            
            # Добавляем информацию о поиске
            result['search_query'] = question
            result['search_results_count'] = len(relevant_docs)
            result['total_processing_time'] = time.time() - total_start_time
            
            logger.info(f"Обработка вопроса завершена за {result['total_processing_time']:.2f}с")
            
            return result
            
        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error(f"Ошибка при обработке вопроса: {e}")
            return {
                'answer': f'Произошла ошибка при обработке вопроса: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'search_query': question,
                'search_results_count': 0,
                'total_processing_time': total_time,
                'error': str(e)
            }

def main():
    """Демонстрация работы RAG пайплайна"""
    try:
        # Инициализация пайплайна
        rag = RAGPipeline()
        
        # Примеры вопросов для тестирования
        test_questions = [
            "Что такое живопись?",
            "Расскажи о скульптуре",
            "Какие бывают виды искусства?",
            "Что такое музейные экспонаты?"
        ]
        
        print("=" * 60)
        print("ДЕМОНСТРАЦИЯ RAG ПАЙПЛАЙНА")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Вопрос: {question}")
            print("-" * 40)
            
            # Получаем ответ
            result = rag.ask_question(question, top_k=3)
            
            # Выводим результат
            print(f"Ответ: {result['answer']}")
            print(f"Уверенность: {result['confidence']:.4f}")
            print(f"Время обработки: {result['total_processing_time']:.2f}с")
            print(f"Найдено источников: {result['total_sources']}")
            
            if result['sources']:
                print("\nИсточники:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"  {j}. {source['source_file']} (релевантность: {source['relevance_score']:.4f})")
                    print(f"     {source['content'][:100]}...")
            
            print("\n" + "=" * 60)
    
    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main() 