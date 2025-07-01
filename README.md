# RAG System с Groq LLM

Система поиска и ответов на вопросы (RAG - Retrieval-Augmented Generation) с использованием Groq API и модели llama3-70b-8192.

## Структура проекта

```
project_new/
├── rag_data_preparation.py    # Подготовка данных
├── create_embeddings_and_store.py  # Создание эмбеддингов
├── rag_pipeline.py            # RAG пайплайн с LLM
├── test_rag.py               # Тестирование системы
├── config.py                 # Конфигурация
├── requirements.txt          # Зависимости
├── data_chunks.json          # Обработанные данные
├── chroma_db/                # Векторная база данных
└── collection-main/          # Исходные данные
```

## Установка и настройка

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка API ключа

Создайте файл `.env` в корневой директории проекта:

```env
GROQ_API_KEY=ваш_ключ_groq_api_здесь
```

Получите API ключ на сайте [Groq](https://console.groq.com/).

### 3. Подготовка данных (если еще не выполнено)

```bash
python rag_data_preparation.py
```

### 4. Создание эмбеддингов (если еще не выполнено)

```bash
python create_embeddings_and_store.py
```

## Использование

### Тестирование RAG пайплайна

```bash
python test_rag.py
```

### Интерактивный режим

```bash
python test_rag.py --interactive
```

### Использование в коде

```python
from rag_pipeline import RAGPipeline

# Инициализация
rag = RAGPipeline()

# Задать вопрос
result = rag.ask_question("Что такое живопись?", top_k=5)

# Получить ответ
print(result['answer'])
print(f"Уверенность: {result['confidence']}")
print(f"Источники: {result['sources']}")
```

## Функциональность

### 1. LLM Setup
- ✅ Подключение к Groq API
- ✅ Использование модели llama3-70b-8192
- ✅ Создание промпт-шаблона для генерации ответов
- ✅ Обработка ошибок и таймаутов

### 2. RAG Pipeline
- ✅ Объединение поиска и генерации в единый пайплайн
- ✅ Функция `ask_question()` для получения ответов с источниками
- ✅ Логирование всех этапов обработки

## API Endpoints

Система предоставляет следующие возможности:

- **Поиск документов**: `search_documents(query, top_k)`
- **Генерация ответов**: `generate_answer(question, context_docs)`
- **Полный RAG процесс**: `ask_question(question, top_k)`

## Логирование

Все операции логируются в файл `rag_system.log` и выводятся в консоль.

## Конфигурация

Основные параметры можно изменить в файле `config.py`:

- `LLM_MODEL`: модель LLM (по умолчанию "llama3-70b-8192")
- `EMBEDDING_MODEL`: модель эмбеддингов
- `DEFAULT_TOP_K`: количество документов для поиска
- `LLM_TEMPERATURE`: температура генерации
- `LLM_MAX_TOKENS`: максимальное количество токенов

## Примеры использования

### Базовый поиск

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
documents = rag.search_documents("живопись", top_k=3)
```

### Получение ответа

```python
result = rag.ask_question("Что такое живопись?")
print(result['answer'])
```

### Анализ результатов

```python
result = rag.ask_question("Расскажи о скульптуре")

print(f"Ответ: {result['answer']}")
print(f"Уверенность: {result['confidence']:.4f}")
print(f"Время обработки: {result['total_processing_time']:.2f}с")
print(f"Источники: {result['total_sources']}")

for source in result['sources']:
    print(f"- {source['source_file']} (релевантность: {source['relevance_score']:.4f})")
```

## Требования

- Python 3.8+
- Groq API ключ
- Доступ к интернету для загрузки моделей эмбеддингов

## Устранение неполадок

1. **Ошибка API ключа**: Убедитесь, что файл `.env` создан и содержит правильный ключ
2. **Ошибка загрузки моделей**: Проверьте подключение к интернету
3. **Ошибка ChromaDB**: Убедитесь, что эмбеддинги были созданы
4. **Таймауты**: Увеличьте `LLM_TIMEOUT` в конфигурации

## Логи

Подробные логи сохраняются в файл `rag_system.log` для отладки. 