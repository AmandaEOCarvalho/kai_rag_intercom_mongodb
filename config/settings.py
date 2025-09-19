import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Intercom
    INTERCOM_API_TOKEN = os.getenv("INTERCOM_API_TOKEN")
    INTERCOM_BASE_URL = os.getenv("INTERCOM_BASE_URL")
    
    # MongoDB
    MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
    DATABASE_NAME = os.getenv("KYTE_DBNAME_AI")
    COLLECTION_NAME = os.getenv("KYTE_COLLECTION_NAME")
    
    # Processing Settings
    ## AI Models
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    RAG_SYNTH_MODEL = os.getenv("RAG_SYNTH_MODEL")
    RAG_IMAGE_PROCESSOR_MODEL = os.getenv("RAG_IMAGE_PROCESSOR_MODEL")
    RAG_CONTEXTUAL_ENRICHER_MODEL = os.getenv("RAG_CONTEXTUAL_ENRICHER_MODEL")
    RAG_CHUNKER_MODEL = os.getenv("RAG_CHUNKER_MODEL")
    RAG_CATEGORIZER_MODEL = os.getenv("RAG_CATEGORIZER_MODEL")
    ## Configs
    MAX_CHUNK_SIZE = os.getenv("MAX_CHUNK_SIZE")
    EMBEDDING_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS")
    
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
