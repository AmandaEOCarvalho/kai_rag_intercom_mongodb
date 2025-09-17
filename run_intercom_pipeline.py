# run_intercom_pipeline.py - Versão Corrigida
import sys
import os

# Adiciona o diretório raiz do projeto ao Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.settings import Config
from src.api.intercom_client import IntercomClient
from src.processing.text_processor import TextProcessor
from src.processing.chunker import LLMChunker
from src.processing.contextual_enricher import ContextualEnricher
from src.processing.categorizer import ArticleCategorizer
from src.utils.embeddings import EmbeddingGenerator
from src.utils.text_cleaner import TextCleaner
from src.mongodb.mongodb_client import MongoDBClient


def list_all_collections(intercom_client: IntercomClient) -> dict:
    """Lista todas as coleções disponíveis na Intercom (para achar a de RAG)."""
    try:
        collections = intercom_client.list_collections()
        print("\n📚 Coleções disponíveis na Intercom:")
        print("-" * 50)
        if collections and "data" in collections:
            for c in collections["data"]:
                print(f"ID: {c.get('id')}")
                print(f"Nome: {c.get('name', '—')}")
                print(f"Descrição: {c.get('description', '—')}")
                print("-" * 30)
        else:
            print("Nenhuma coleção encontrada.")
        return collections
    except Exception as e:
        print(f"❌ Erro ao listar coleções: {e}")
        return {}


def is_rag_eligible_article(article: dict, rag_collection_id: str = None, excluded_article_ids: list = None) -> bool:
    """Determina se um artigo é elegível para o RAG baseado na coleção e exclusões."""
    article_id = str(article.get("id", ""))

    if excluded_article_ids and article_id in excluded_article_ids:
        return False

    if rag_collection_id:
        parent_ids = article.get("parent_ids", [])
        if rag_collection_id not in [str(pid) for pid in parent_ids]:
            return False

    translated_content = article.get("translated_content", {})
    if not translated_content:
        return False

    for _, content in translated_content.items():
        if isinstance(content, dict) and content.get("body"):
            state = content.get("state", "")
            if state == "published" or (state == "draft" and rag_collection_id):
                return True

    return False


def process_single_article(article: dict, components: dict, rag_collection_id: str = None, excluded_article_ids: list = None) -> list:
    """
    Processa um único artigo da Intercom seguindo as melhores práticas:
    HTML → Markdown → Categorização → Chunking → Contextual Enrichment → Limpeza Condicional → Embeddings
    """
    article_id = article.get("id")
    documents_for_db = []

    if not is_rag_eligible_article(article, rag_collection_id, excluded_article_ids):
        if excluded_article_ids and str(article_id) in excluded_article_ids:
            print(f" -> Artigo {article_id} pulado: está na lista de exclusões.")
        else:
            print(f" -> Artigo {article_id} pulado: não está na coleção RAG ou não tem conteúdo válido.")
        return documents_for_db

    for lang, content in article.get("translated_content", {}).items():
        if not (isinstance(content, dict) and content.get("body")):
            continue

        state = content.get("state", "")
        if not (state == "published" or (state == "draft" and rag_collection_id)):
            continue

        print(f"\n📄 Processando Artigo ID: {article_id}, Idioma: {lang}, Estado: {state}")

        # ✅ ETAPA 1: HTML → Markdown formatado (preserva estrutura semântica)
        html_body = content["body"]
        markdown_text = components["text_processor"].process_html_body(html_body)
        if not markdown_text:
            print(" -> Artigo pulado: sem texto após parsing HTML.")
            continue

        print(f"📝 Markdown gerado: {len(markdown_text)} chars")

        # ✅ ETAPA 2: Categorização (usa markdown formatado)
        title = content.get("title", f"Artigo {article_id}")
        category = components["categorizer"].categorize_article(markdown_text, title)
        print(f" -> Categoria identificada: {category}")

        # ✅ ETAPA 3: Chunking semântico (usa markdown formatado)
        print(" -> Iniciando chunking semântico...")
        chunks = components["chunker"].chunk_text(markdown_text)
        print(f" -> Gerados {len(chunks)} chunks semânticos")

        # ✅ ETAPA 4: Contextual Enrichment (usa markdown formatado)
        print(" -> Iniciando enriquecimento contextual...")
        enriched_chunks = components["enricher"].enrich_chunks(chunks, markdown_text)
        print(f" -> Enriquecidos {len(enriched_chunks)} chunks com contexto")

        # ✅ ETAPA 5: Limpeza condicional + Embeddings (APENAS agora limpa para vetor)
        text_cleaner = components["text_cleaner"]
        
        for i, contextualized_chunk in enumerate(enriched_chunks):
            print(f" -> Processando chunk {i+1}/{len(enriched_chunks)}...")
            
            # Limpeza condicional inteligente
            clean_content = text_cleaner.clean_contextual_chunk(contextualized_chunk)
            
            if not clean_content:
                print(f"   ⚠️ Chunk {i+1} vazio após limpeza, pulando.")
                continue

            # Embedding com título + conteúdo limpo
            embedding_input = f"{title}\n\n{clean_content}"
            embedding = components["embedding_generator"].generate(embedding_input)
            
            if not embedding:
                print(f"   ❌ Falha ao gerar embedding para chunk {i+1}")
                continue

            print(f"   ✅ Chunk {i+1} processado com sucesso")

            # Documento final para MongoDB
            document = {
                "title": title,
                "content": clean_content,  # Conteúdo otimizado para embeddings
                "category": category,
                "language": lang,
                "embedding": embedding,
                "meta_data": {
                    "source_type": "intercom_article",
                    "article_id": str(article_id),
                    "intercom_url": content.get("url", ""),
                    "intercomCreatedAt": article.get("created_at"),
                    "intercomUpdatedAt": article.get("updated_at"),
                    "article_state": state,
                    "rag_collection_id": rag_collection_id,
                    "is_chunked": True,
                    "chunk_index": i,
                    "total_chunks": len(enriched_chunks),
                    "embedding_model": Config.EMBEDDING_MODEL,
                    "dimensions": Config.EMBEDDING_DIMENSIONS,
                    "embedding_input": "title+content",
                    "processing_pipeline": "html->markdown->categorize->chunk->enrich->clean->embed"
                }
            }
            documents_for_db.append(document)

        print(f"✅ Artigo {article_id} processado: {len([d for d in documents_for_db if d['meta_data']['article_id'] == str(article_id)])} documentos gerados")

    return documents_for_db


def fetch_all_articles_from_collection(intercom_client: IntercomClient, collection_id: str = None) -> dict:
    """Busca TODOS os artigos da Intercom com paginação completa (opcionalmente por coleção)."""
    all_articles = []
    page = 1
    per_page = 50

    if collection_id:
        print(f"🔍 Buscando TODOS os artigos da coleção ID: {collection_id}")
    else:
        print("🔍 Buscando TODOS os artigos (sem filtro de coleção)")

    while True:
        print(f"📄 Processando página {page}...")
        data = (
            intercom_client.fetch_articles_from_collection(collection_id, page, per_page)
            if collection_id else
            intercom_client.fetch_articles(page, per_page)
        )

        if not data or "data" not in data or not data["data"]:
            print(f"   → Página {page} vazia ou sem dados. Finalizando busca.")
            break

        articles_in_page = len(data["data"])
        all_articles.extend(data["data"])
        print(f"   → Encontrados {articles_in_page} artigos na página {page}")

        if articles_in_page < per_page:
            print(f"   → Última página detectada (menos de {per_page} artigos)")
            break

        page += 1

    total_found = len(all_articles)
    print(f"🎯 Total de artigos coletados: {total_found}")
    return {"data": all_articles}


def main():
    """
    Pipeline principal seguindo as melhores práticas do tutorial:
    - Preserva markdown formatado até o enriquecimento contextual
    - Aplica limpeza condicional apenas para embeddings
    - Usa módulo unificado de limpeza de texto
    """
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Erro de configuração: {e}")
        return

    # ✅ Inicializa componentes incluindo o novo TextCleaner
    components = {
        "intercom_client": IntercomClient(),
        "text_processor": TextProcessor(),           # Agora preserva markdown
        "chunker": LLMChunker(),
        "enricher": ContextualEnricher(),
        "categorizer": ArticleCategorizer(),
        "embedding_generator": EmbeddingGenerator(),
        "text_cleaner": TextCleaner(),              # ✅ Novo componente unificado
        "mongodb_client": MongoDBClient()
    }

    print("🚀 Iniciando pipeline de processamento de artigos da Intercom...")
    print("📋 Pipeline: HTML → Markdown → Categorizar → Chunking → Enriquecimento → Limpeza → Embeddings")

    # Configurações
    RAG_COLLECTION_ID = None  # ex.: "123456" para filtrar coleção específica
    EXCLUDED_ARTICLE_IDS = ["7861154"]  # IDs a excluir

    # (Opcional) Listar coleções para encontrar a ID correta
    # list_all_collections(components["intercom_client"])

    # Busca artigos
    intercom_data = fetch_all_articles_from_collection(
        components["intercom_client"],
        collection_id=RAG_COLLECTION_ID
    )

    if not intercom_data or "data" not in intercom_data:
        print("⚠️ Nenhum artigo da Intercom encontrado para processar.")
        return

    print(f"📊 Total de artigos encontrados: {len(intercom_data['data'])}")

    # Processa artigos
    all_processed_documents = []
    processed_count = 0
    skipped_count = 0

    for article in intercom_data["data"]:
        processed_docs = process_single_article(
            article,
            components,
            RAG_COLLECTION_ID,
            EXCLUDED_ARTICLE_IDS
        )
        if processed_docs:
            all_processed_documents.extend(processed_docs)
            processed_count += 1
        else:
            skipped_count += 1

    # Relatório final
    print(f"\n📈 Resumo do processamento:")
    print(f" • Artigos processados: {processed_count}")
    print(f" • Artigos pulados: {skipped_count}")
    print(f" • Total de documentos gerados: {len(all_processed_documents)}")

    # Salva no MongoDB
    if all_processed_documents:
        print(f"\n💾 Salvando {len(all_processed_documents)} documentos no MongoDB...")
        components["mongodb_client"].upsert_documents(all_processed_documents)
        print("✅ Documentos salvos com sucesso!")
    else:
        print("❌ Nenhum documento foi gerado a partir dos artigos da Intercom.")

    print("\n🎉 Pipeline de artigos da Intercom concluído!")
    print("📋 Processo seguiu as melhores práticas: markdown preservado até limpeza final para embeddings")


if __name__ == "__main__":
    main()