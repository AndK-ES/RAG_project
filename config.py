import os
from dotenv import load_dotenv
import logging

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Конфигурация API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY не найден в переменных окружения. Создайте файл .env с вашим ключом API")
    raise ValueError("GROQ_API_KEY не найден в переменных окружения")

# Конфигурация модели
LLM_MODEL = "llama3-70b-8192"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Конфигурация базы данных
CHROMA_DB_DIR = "chroma_db"

# Конфигурация поиска
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Конфигурация LLM
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 30  # секунды

logger.info("Конфигурация загружена успешно") 