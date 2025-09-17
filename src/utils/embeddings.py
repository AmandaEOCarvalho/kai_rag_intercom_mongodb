from openai import OpenAI
from config.settings import Config

class EmbeddingGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.EMBEDDING_MODEL
        self.dimensions = Config.EMBEDDING_DIMENSIONS
    
    def generate(self, text: str) -> list:
        """Gera embedding para um texto"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Erro ao gerar embedding: {e}")
            return []