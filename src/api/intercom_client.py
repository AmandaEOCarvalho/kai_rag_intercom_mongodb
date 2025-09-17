import requests
from typing import Dict, Optional
from config.settings import Config

class IntercomClient:
    def __init__(self):
        self.base_url = Config.INTERCOM_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {Config.INTERCOM_API_TOKEN}",
            "Accept": "application/json"
        }
    
    def fetch_articles(self, page_number: int = 1, per_page: int = 10) -> Optional[Dict]:
        """Busca artigos da API do Intercom (método original)"""
        url = f"{self.base_url}/articles"
        params = {
            "page": page_number, 
            "per_page": per_page
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar dados da API do Intercom: {e}")
            return None

    def list_collections(self) -> Optional[Dict]:
        """
        Lista todas as coleções da Central de Ajuda da Intercom.
        
        Returns:
            Optional[Dict]: Resposta da API com todas as coleções ou None em caso de erro
        """
        url = f"{self.base_url}/help_center/collections"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao listar coleções: {e}")
            return None

    def fetch_articles_from_collection(self, collection_id: str, page_number: int = 1, per_page: int = 50) -> Optional[Dict]:
        """
        Busca artigos de uma coleção específica da Intercom.
        
        Args:
            collection_id (str): ID da coleção
            page_number (int): Número da página (padrão: 1)
            per_page (int): Artigos por página (padrão: 50)
            
        Returns:
            Optional[Dict]: Resposta da API com os artigos da coleção ou None em caso de erro
        """
        url = f"{self.base_url}/help_center/collections/{collection_id}/articles"
        params = {
            "page": page_number,
            "per_page": per_page
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao buscar artigos da coleção {collection_id}: {e}")
            return None

    def fetch_all_articles_including_drafts(self, page_number: int = 1, per_page: int = 50) -> Optional[Dict]:
        """
        Busca todos os artigos da Intercom, incluindo rascunhos.
        
        Args:
            page_number (int): Número da página (padrão: 1)
            per_page (int): Artigos por página (padrão: 50)
            
        Returns:
            Optional[Dict]: Resposta da API com todos os artigos ou None em caso de erro
        """
        url = f"{self.base_url}/articles"
        params = {
            "page": page_number,
            "per_page": per_page,
            "include_draft_articles": "true"  # Inclui artigos em rascunho
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao buscar artigos (incluindo rascunhos): {e}")
            return None

    def get_collection_details(self, collection_id: str) -> Optional[Dict]:
        """
        Busca detalhes de uma coleção específica.
        
        Args:
            collection_id (str): ID da coleção
            
        Returns:
            Optional[Dict]: Detalhes da coleção ou None em caso de erro
        """
        url = f"{self.base_url}/help_center/collections/{collection_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao buscar detalhes da coleção {collection_id}: {e}")
            return None

    def fetch_article_by_id(self, article_id: str) -> Optional[Dict]:
        """
        Busca um artigo específico pelo ID.
        
        Args:
            article_id (str): ID do artigo
            
        Returns:
            Optional[Dict]: Dados do artigo ou None em caso de erro
        """
        url = f"{self.base_url}/articles/{article_id}"
        params = {
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao buscar artigo {article_id}: {e}")
            return None

    def search_articles(self, query: str, collection_id: str = None) -> Optional[Dict]:
        """
        Busca artigos usando a funcionalidade de busca da Intercom.
        
        Args:
            query (str): Termo de busca
            collection_id (str, optional): ID da coleção para filtrar
            
        Returns:
            Optional[Dict]: Resultados da busca ou None em caso de erro
        """
        url = f"{self.base_url}/articles/search"
        params = {
            "phrase": query
        }
        
        if collection_id:
            params["collection_id"] = collection_id
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao buscar artigos com query '{query}': {e}")
            return None

    def test_connection(self) -> bool:
        """
        Testa a conexão com a API da Intercom.
        
        Returns:
            bool: True se a conexão estiver funcionando, False caso contrário
        """
        url = f"{self.base_url}/me"  # Endpoint para verificar autenticação
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            print("✅ Conexão com a API da Intercom estabelecida com sucesso")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro de conexão com a API da Intercom: {e}")
            return False