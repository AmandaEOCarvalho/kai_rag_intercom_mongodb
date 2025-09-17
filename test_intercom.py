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
from src.utils.text_cleaner import TextCleaner  # ✅ Novo módulo unificado
# ✅ NOVA CONFIGURAÇÃO: IDs que devem ter todos os idiomas (PT, EN, ES)
MULTILINGUAL_ARTICLE_IDS = [
    "7861149", "7915496", "8411647", "8887223", "7915619",
    "7861109", "10008263", "7885145", "7992438", "7914908"
]

# Função igual ao pipeline principal
def get_allowed_languages(article_id: str, multilingual_article_ids: list) -> list:
    """
    Determina quais idiomas processar baseado no ID do artigo.
    - Para IDs específicos: processa PT, EN, ES
    - Para demais artigos: apenas PT-BR
    """
    if str(article_id) in multilingual_article_ids:
        return ["pt", "pt-BR", "en", "es"]  # Todos os idiomas para artigos específicos
    else:
        return ["pt", "pt-BR"]  # Aceita pt e pt-BR para os demais


def is_rag_eligible_article(article: dict, rag_collection_id: str = None, excluded_article_ids: list = None) -> bool:
    """Determina se um artigo é elegível para o RAG baseado na coleção e lista de exclusões."""
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


def process_single_article_test(article: dict, components: dict, rag_collection_id: str = None, excluded_article_ids: list = None) -> list:
    """
    Versão de teste que processa um artigo seguindo as melhores práticas
    mas NÃO salva no MongoDB. Mostra todo o pipeline em ação.
    """
    article_id = article.get("id")
    documents_for_db = []

    if not is_rag_eligible_article(article, rag_collection_id, excluded_article_ids):
        if excluded_article_ids and str(article_id) in excluded_article_ids:
            print(f" -> Artigo {article_id} pulado: está na lista de exclusões.")
        else:
            print(f" -> Artigo {article_id} pulado: não está na coleção RAG ou não tem conteúdo válido.")
        return documents_for_db


    # Se for coleção RAG, processa todos os idiomas disponíveis
    if rag_collection_id:
        allowed_languages = list(article.get("translated_content", {}).keys())
        print(f"📋 Artigo {article_id} (coleção RAG) - Todos idiomas permitidos: {allowed_languages}")
    else:
        allowed_languages = get_allowed_languages(article_id, MULTILINGUAL_ARTICLE_IDS)
        print(f"📋 Artigo {article_id} - Idiomas permitidos: {allowed_languages}")

    for lang, content in article.get("translated_content", {}).items():
        # Se for coleção RAG, não filtra idiomas
        if not rag_collection_id and lang not in allowed_languages:
            print(f" -> Idioma {lang} pulado para artigo {article_id} (não está na lista permitida)")
            continue

        if not (isinstance(content, dict) and content.get("body")):
            continue

        state = content.get("state", "")
        if not (state == "published" or (state == "draft" and rag_collection_id)):
            continue

        print(f"\n📄 TESTANDO Artigo ID: {article_id}, Idioma: {lang}, Estado: {state}")
        title = content.get("title", "Sem título")
        print(f"📝 Título: {title}")

        # ✅ ETAPA 1: HTML → Markdown formatado (preserva estrutura)
        print("\n🔧 ETAPA 1: Convertendo HTML → Markdown formatado...")
        html_body = content["body"]
        markdown_text = components["text_processor"].process_html_body(html_body)
        
        if not markdown_text:
            print(" -> Artigo pulado pois não contém texto após o parsing.")
            continue

        print(f"📏 Markdown gerado: {len(markdown_text)} chars")
        print(f"🔤 Prévia do markdown (400 chars):")
        print(f"{markdown_text[:400]}...")
        print("=" * 50)

        # ✅ ETAPA 2: Categorização (usa markdown formatado)
        print("\n🏷️  ETAPA 2: Categorizando com markdown formatado...")
        category = components["categorizer"].categorize_article(markdown_text, title)
        print(f"🏷️  Categoria identificada: {category}")

        # ✅ ETAPA 3: Chunking semântico (usa markdown formatado)
        print("\n✂️  ETAPA 3: Chunking semântico com markdown...")
        chunks = components["chunker"].chunk_text(markdown_text)
        print(f"✂️  Gerados {len(chunks)} chunks semânticos")
        
        if chunks:
            print(f"\n📋 EXEMPLO DE CHUNK SEMÂNTICO (bruto, com markdown):")
            print(f"Tamanho: {len(chunks[0])} chars")
            print(chunks[0][:500] + ("..." if len(chunks[0]) > 500 else ""))
            print("-" * 70)

        # ✅ ETAPA 4: Enriquecimento contextual (usa markdown formatado)
        print("\n🔧 ETAPA 4: Enriquecimento contextual...")
        enriched_chunks = components["enricher"].enrich_chunks(chunks, markdown_text)
        print(f"🔧 Gerados e enriquecidos {len(enriched_chunks)} chunks finais")

        # Mostra exemplo de chunk enriquecido (ainda com markdown)
        if enriched_chunks:
            print(f"\n📋 EXEMPLO DE CHUNK ENRIQUECIDO (ainda com markdown):")
            print(f"Tamanho: {len(enriched_chunks[0])} chars")
            print(enriched_chunks[0][:600] + ("..." if len(enriched_chunks[0]) > 600 else ""))
            print("-" * 70)

        # ✅ ETAPA 5: Limpeza condicional + Embeddings (APENAS agora limpa)
        print(f"\n🧽 ETAPA 5: Limpeza condicional + Embeddings...")
        text_cleaner = components["text_cleaner"]
        
        for i, contextualized_chunk in enumerate(enriched_chunks):
            print(f"\n🧽 Processando chunk {i+1}/{len(enriched_chunks)}...")
            
            # Mostra estado antes da limpeza
            print(f"📦 ANTES da limpeza (chunk {i+1}) — len={len(contextualized_chunk)}:")
            print(contextualized_chunk[:300] + ("..." if len(contextualized_chunk) > 300 else ""))
            
            # Verifica se precisa de limpeza
            needs_cleaning = text_cleaner.looks_like_markdown_or_html(contextualized_chunk)
            print(f"🔍 Precisa de limpeza estrutural? {'✅ SIM' if needs_cleaning else '❌ NÃO (apenas normalização)'}")
            
            # Aplica limpeza condicional
            clean_content = text_cleaner.clean_contextual_chunk(contextualized_chunk)
            
            if not clean_content:
                print(f"   ⚠️ Chunk {i+1} vazio após limpeza, pulando.")
                continue

            # Mostra resultado da limpeza
            print(f"\n📦 DEPOIS da limpeza (chunk {i+1}) — len={len(clean_content)}:")
            print(clean_content[:300] + ("..." if len(clean_content) > 300 else ""))

            # Input para embedding
            embedding_input = f"{title}\n\n{clean_content}"
            print(f"\n🧠 INPUT para embedding (título + conteúdo limpo) — len={len(embedding_input)}:")
            print(embedding_input[:400] + ("..." if len(embedding_input) > 400 else ""))

            # Gerar embedding
            print(f"\n🧠 Gerando embedding para chunk {i+1}...")
            embedding = components["embedding_generator"].generate(embedding_input)
            if not embedding:
                print(f"   ❌ Falha ao gerar embedding para chunk {i+1}")
                continue

            print(f"   ✅ Embedding gerado: {len(embedding)} dimensões")

            print(f"\n📋 CONTEÚDO FINAL OTIMIZADO (chunk {i+1}):")
            print("=" * 70)
            print(clean_content)
            print("=" * 70)

            # Documento simulado (sem salvar)
            document = {
                "title": title,
                "content": clean_content,
                "category": category,
                "language": lang,
                "embedding": f"[EMBEDDING COM {len(embedding)} DIMENSÕES]",
                "meta_data": {
                    "source_type": "intercom_article",
                    "article_id": str(article_id),
                    "intercom_url": content.get("url", ""),
                    "article_state": state,
                    "rag_collection_id": rag_collection_id,
                    "is_chunked": True,
                    "chunk_index": i,
                    "total_chunks": len(enriched_chunks),
                    "embedding_model": Config.EMBEDDING_MODEL,
                    "dimensions": Config.EMBEDDING_DIMENSIONS,
                    "cleaned_conditionally": needs_cleaning
                }
            }
            documents_for_db.append(document)

        print(f"\n✅ Artigo {article_id} processado com sucesso!")
        print("=" * 80)

    return documents_for_db


def main():
    """
    Versão de teste para processar apenas 3 artigos seguindo as melhores práticas.
    Mostra todo o pipeline em ação sem salvar no MongoDB.
    """
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Erro de configuração: {e}")
        return

    # ✅ Inicializar componentes incluindo o novo TextCleaner
    components = {
        "intercom_client": IntercomClient(),
        "text_processor": TextProcessor(),           # Agora preserva markdown
        "chunker": LLMChunker(),
        "enricher": ContextualEnricher(),
        "categorizer": ArticleCategorizer(),
        "embedding_generator": EmbeddingGenerator(),
        "text_cleaner": TextCleaner()               # ✅ Novo componente unificado
    }

    print("🧪 MODO TESTE - Processando apenas 3 artigos (SEM salvar no MongoDB)")
    print("📋 Pipeline: HTML → Markdown → Categorizar → Chunking → Enriquecimento → Limpeza → Embeddings")
    print("=" * 80)

    # Configuração
    RAG_COLLECTION_ID = None  # Ajuste se quiser filtrar por coleção específica
    EXCLUDED_ARTICLE_IDS = ["7861154"]  # Exemplo de exclusão

    # Busca primeiros 3 artigos
    print("🔍 Buscando primeiros 3 artigos...")
    intercom_data = components["intercom_client"].fetch_articles(page_number=1, per_page=3)

    if not intercom_data or "data" not in intercom_data:
        print("⚠️ Nenhum artigo encontrado.")
        return

    print(f"📊 Artigos encontrados para teste: {len(intercom_data['data'])}")

    all_processed_documents = []
    processed_count = 0
    skipped_count = 0

    for i, article in enumerate(intercom_data["data"]):
        print(f"\n🎯 PROCESSANDO ARTIGO {i+1}/3")
        processed_docs = process_single_article_test(
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
    print(f"\n📈 RESUMO DO TESTE")
    print("-" * 80)
    print(f" • Artigos processados: {processed_count}")
    print(f" • Artigos pulados: {skipped_count}")
    print(f" • Total de documentos gerados: {len(all_processed_documents)}")
    print(f" • MongoDB: NÃO UTILIZADO (modo teste)")

    if all_processed_documents:
        print("\n📋 EXEMPLO DE DOCUMENTO FINAL (campos principais):")
        example_doc = all_processed_documents[0]
        print(f"   Título: {example_doc['title']}")
        print(f"   Categoria: {example_doc['category']}")
        print(f"   Idioma: {example_doc['language']}")
        print(f"   Tamanho do conteúdo: {len(example_doc['content'])} chars")
        print(f"   Pipeline usado: {example_doc['meta_data']['processing_pipeline']}")
        print(f"   Limpeza aplicada: {example_doc['meta_data']['cleaned_conditionally']}")

    print("\n🎉 Teste concluído! Pipeline seguiu as melhores práticas:")
    print("   ✅ HTML convertido para markdown formatado")
    print("   ✅ Markdown preservado durante categorização e chunking")
    print("   ✅ Contextual enrichment trabalhou com markdown")
    print("   ✅ Limpeza condicional aplicada apenas para embeddings")
    print("   ✅ Conteúdo final otimizado para busca vetorial")


if __name__ == "__main__":
    main()