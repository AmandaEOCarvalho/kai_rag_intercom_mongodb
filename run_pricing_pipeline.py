from config.settings import Config
from src.api.kyte_client import generate_pricing_documents_from_api
from src.utils.embeddings import EmbeddingGenerator
from src.mongodb.mongodb_client import MongoDBClient

def update_pricing_knowledge():
    """
    Pipeline dedicado a buscar, processar e salvar os documentos
    de preços da Kyte.
    """
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Erro de configuração: {e}")
        return

    # Inicializar os componentes necessários
    embedding_generator = EmbeddingGenerator()
    mongodb_client = MongoDBClient()

    print("🚀 Iniciando pipeline de atualização de preços...")
    
    all_pricing_documents = []

    # ETAPA 1: Gerar documentos a partir da API da Kyte
    pricing_documents_raw = generate_pricing_documents_from_api()

    if pricing_documents_raw:
        for doc in pricing_documents_raw:
            embedding_text = f"{doc['title']}. {doc['content']}"
            embedding = embedding_generator.generate(embedding_text)
            
            if embedding:
                doc['embedding'] = embedding
                doc['meta_data']['embedding_model'] = Config.EMBEDDING_MODEL
                doc['meta_data']['dimensions'] = Config.EMBEDDING_DIMENSIONS
                all_pricing_documents.append(doc)
    else:
        print("⚠️ Nenhum documento de preço gerado pela API.")
    
    # ETAPA 2: Salvar os documentos no MongoDB
    if all_pricing_documents:
        print(f"\n💾 Total de {len(all_pricing_documents)} documentos de preço para salvar.")
        mongodb_client.upsert_documents(all_pricing_documents)
    else:
        print("❌ Nenhum documento de preço foi processado.")

    print("\n🎉 Pipeline de preços concluído!")

if __name__ == "__main__":
    update_pricing_knowledge()