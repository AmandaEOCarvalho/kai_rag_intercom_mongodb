import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Intercom
    INTERCOM_API_TOKEN = os.getenv("INTERCOM_API_TOKEN")
    INTERCOM_BASE_URL = "https://api.intercom.io"
    
    # MongoDB
    MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
    DATABASE_NAME = os.getenv("KYTE_DBNAME_AI", "kyte-ai")
    COLLECTION_NAME = "KyteFAQKnowledgeBase"
    
    # Processing Settings
    MAX_CHUNK_SIZE = 2000
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536
    
    @classmethod
    def validate(cls):
        """Valida se todas as configurações necessárias estão presentes"""
        required = [
            cls.OPENAI_API_KEY,
            cls.INTERCOM_API_TOKEN,
            cls.MONGODB_CONNECTION_STRING
        ]
        missing = [name for name, value in zip(
            ["OPENAI_API_KEY", "INTERCOM_API_TOKEN", "MONGODB_CONNECTION_STRING"],
            required
        ) if not value]
        
        if missing:
            raise ValueError(f"Configurações faltando: {', '.join(missing)}")
