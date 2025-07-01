#!/usr/bin/env python3
"""
Тестовый скрипт для RAG пайплайна
"""

import os
from rag_pipeline import RAGPipeline
import config

def test_rag_pipeline():
    """Тестирование RAG пайплайна"""
    print("🚀 Запуск тестирования RAG пайплайна")
    print("=" * 50)
    
    try:
        # Проверяем наличие API ключа
        if not config.GROQ_API_KEY:
            print("❌ Ошибка: GROQ_API_KEY не найден в переменных окружения")
            print("Создайте файл .env с содержимым: GROQ_API_KEY=ваш_ключ_здесь")
            return
        
        print("✅ API ключ найден")
        
        # Инициализация RAG пайплайна
        print("🔄 Инициализация RAG пайплайна...")
        rag = RAGPipeline()
        print("✅ RAG пайплайн инициализирован")
        
        # Тестовые вопросы
        test_questions = [
            "Что такое живопись?",
            "Расскажи о скульптуре",
            "Какие бывают виды искусства?",
            "Что такое музейные экспонаты?"
        ]
        
        print("\n" + "=" * 50)
        print("ТЕСТИРОВАНИЕ ВОПРОСОВ")
        print("=" * 50)
        
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
            
            print("\n" + "=" * 50)
        
        print("\n✅ Тестирование завершено успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        print("Проверьте логи в файле rag_system.log")

def interactive_mode():
    """Интерактивный режим для тестирования"""
    print("🎯 ИНТЕРАКТИВНЫЙ РЕЖИМ ТЕСТИРОВАНИЯ")
    print("Введите 'quit' для выхода")
    print("=" * 50)
    
    try:
        rag = RAGPipeline()
        
        while True:
            question = input("\nВведите ваш вопрос: ").strip()
            
            if question.lower() == 'quit':
                print("До свидания!")
                break
            
            if not question:
                print("Пожалуйста, введите вопрос")
                continue
            
            print(f"\n🔍 Обработка вопроса: '{question}'")
            print("-" * 40)
            
            result = rag.ask_question(question, top_k=5)
            
            print(f"Ответ: {result['answer']}")
            print(f"Уверенность: {result['confidence']:.4f}")
            print(f"Время обработки: {result['total_processing_time']:.2f}с")
            
            if result['sources']:
                print(f"\nИсточники ({result['total_sources']}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['source_file']} (релевантность: {source['relevance_score']:.4f})")
            
            print("-" * 40)
    
    except Exception as e:
        print(f"❌ Ошибка в интерактивном режиме: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        test_rag_pipeline() 