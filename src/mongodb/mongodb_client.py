from pymongo import MongoClient, UpdateOne
from typing import List, Dict
from config.settings import Config

class MongoDBClient:
    def __init__(self):
        self.connection_string = Config.MONGODB_CONNECTION_STRING
        self.database_name = Config.DATABASE_NAME
        self.collection_name = Config.COLLECTION_NAME
        self.client = None
        self.collection = None

    def connect(self):
        """Estabelece conexão com MongoDB e define a coleção."""
        if not self.client:
            self.client = MongoClient(self.connection_string)
            db = self.client[self.database_name]
            self.collection = db[self.collection_name]
        return self.collection

    def close_connection(self):
        """Fecha a conexão com o MongoDB."""
        if self.client:
            self.client.close()
            self.client = None
            print("Conexão com o MongoDB fechada.")

    def upsert_documents(self, documents: List[Dict]) -> None:
        """
        Faz o upsert dos documentos processados para a coleção KyteFAQKnowledgeBase no MongoDB.
        """
        if not documents:
            print("⚠️ Nenhum documento para salvar no MongoDB.")
            return

        try:
            collection = self.connect()

            # (O Bloco de Debug 1 pode ser removido se você não precisar mais dele)
            print("\n--- 🕵️‍♂️ Verificando Configurações do MongoDB ---")
            if self.connection_string and len(self.connection_string) > 30:
                 print(f"CONEXÃO: {self.connection_string[:20]}...{self.connection_string[-10:]}")
            else:
                 print("CONEXÃO: (String de conexão inválida ou curta)")
            print(f"BANCO DE DADOS: {self.database_name}")
            print(f"COLEÇÃO: {collection.name}")
            print("-------------------------------------------\n")

            print(f"Iniciando upsert de {len(documents)} documentos...")
            operations = []
            for doc in documents:
                meta = doc.get("meta_data", {})
                filter_query = {
                    "meta_data.article_id": meta.get("article_id"),
                    "meta_data.language": doc.get("language"),
                    "meta_data.chunk_index": meta.get("chunk_index")
                }
                update_operation = UpdateOne(filter_query, {"$set": doc}, upsert=True)
                operations.append(update_operation)

            if operations:
                result = collection.bulk_write(operations)
                
                # --- CÓDIGO CORRIGIDO AQUI ---
                # Acessamos o dicionário da API diretamente, que é mais seguro
                api_result = result.bulk_api_result
                print(" -> Operação de Bulk Write enviada.")
                print(f" -> Resultado Completo da API: {api_result}")

                # Verificamos se a chave 'writeErrors' existe e não está vazia
                if api_result.get('writeErrors'):
                    print("\n❌ ERROS DE ESCRITA ENCONTRADOS PELO MONGODB:")
                    for error in api_result['writeErrors']:
                        print(f"  - Índice: {error.get('index')}, Código: {error.get('code')}, Mensagem: {error.get('errmsg')}")
                # --- FIM DA CORREÇÃO ---

                print("\n ✅ --- Resumo da Operação ---")
                # Usamos os valores do resultado da API para o log
                print(f" -> {api_result.get('nUpserted', 0)} documentos inseridos (upsert).")
                print(f" -> {api_result.get('nModified', 0)} documentos atualizados.")
        
        except Exception as e:
            print(f"❌ Erro CRÍTICO durante a operação com o MongoDB: {e}")
        finally:
            self.close_connection()