# intercom_pipeline.py
import sys
import os
import re

# Adiciona o diretório raiz do projeto ao Python path para que as importações de 'src' funcionem
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.settings import Config
from src.api.intercom_client import IntercomClient
from src.processing.text_processor import TextProcessor
from src.processing.chunker import LLMChunker
from src.processing.contextual_enricher import ContextualEnricher
from src.processing.categorizer import ArticleCategorizer
from src.utils.embeddings import EmbeddingGenerator
from src.mongodb.mongodb_client import MongoDBClient


# ------------- Limpeza “inteligente” para embeddings -------------

_EMOJI_RE = re.compile(
    "["                                 # principais blocos de emoji/pictogramas
    "\U0001F600-\U0001F64F"             # emoticons
    "\U0001F300-\U0001F5FF"             # símbolos & pictogramas
    "\U0001F680-\U0001F6FF"             # transportes & mapas
    "\U0001F1E0-\U0001F1FF"             # bandeiras
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]",
    flags=re.UNICODE
)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

HTML_TAGS_RE = re.compile(r"<[^>]+>")
MD_CODE_FENCE_RE = re.compile(r"```.+?```", flags=re.DOTALL)
MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
MD_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_SINGLE_ASTERISK_LINE_RE = re.compile(r"(?m)^\s*\*\s*$")
MD_HR_RE = re.compile(r"(?m)^\s{0,3}[-*_]{3,}\s*$")
MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
MD_ITALIC_RE = re.compile(r"\*([^*]+)\*|_([^_]+)_")
_HRULE_RE = re.compile(r"(?m)^\s*(?:\*[\s*]{2,}|[-_]{3,})\s*$")   # *** , * * * , --- , ___
EXTRANEOUS_TOKENS_RE = re.compile(r"(?:\u200b|\u200c|\u200d|\uFEFF)")  # zero-width etc.

def looks_like_html_or_markdown(text: str) -> bool:
    """Heurística leve para decidir quando LIMPAR."""
    if HTML_TAGS_RE.search(text):
        return True
    # marcadores típicos de MD / tokens visuais
    md_markers = ("**", "__", "`", "```", "[", "](", "# ", "\n---", "\r\n---")
    return any(m in text for m in md_markers)

def minimal_normalize(text: str) -> str:
    """Quando NÃO limpar: só normaliza espaços/quebras sem perder semântica."""
    # remove caracteres de controle (mantém \n)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = EXTRANEOUS_TOKENS_RE.sub("", text)
    # colapsa espaços em excesso preservando quebras simples
    text = re.sub(r"[^\S\r\n]+", " ", text)          # múltiplos espaços -> 1
    text = re.sub(r"\n{3,}", "\n\n", text)           # 3+ quebras -> 2
    return text.strip()

def strip_markdown_and_html(text: str) -> str:
    """Remove HTML/Markdown mantendo o conteúdo semântico."""
    # HTML
    text = HTML_TAGS_RE.sub(" ", text)

    # Markdown
    text = MD_CODE_FENCE_RE.sub(" ", text)           # remove blocos ```code```
    text = MD_INLINE_CODE_RE.sub(r"\1", text)        # `inline` -> inline
    text = MD_HEADING_RE.sub("", text)               # remove marcadores de heading
    text = MD_HR_RE.sub(" ", text)                   # ---- -> espaço
    text = _SINGLE_ASTERISK_LINE_RE.sub("", text)           
    text = _HRULE_RE.sub(" ", text)                  
    text = MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)

    # bullets: preserva como “- ” (se existirem)
    text = re.sub(r"(?m)^\s*[\*\•]\s+", "- ", text)

    return text

def clean_for_embedding(text: str) -> str:
    """
    LIMPA quando necessário:
      - remove HTML/Markdown/artefatos
      - remove emojis e caracteres de controle
      - normaliza espaços/quebras
    Mantém acentos, 'R$', '%', e-mails, @, etc.
    """
    # remove emojis & controles
    text = _EMOJI_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = EXTRANEOUS_TOKENS_RE.sub("", text)

    # tira HTML/Markdown se houver
    if looks_like_html_or_markdown(text):
        text = strip_markdown_and_html(text)

    # normaliza espaços/quebras (mantém no máx. 2 quebras)
    text = re.sub(r"[^\S\r\n]+", " ", text)   # colapsa espaços
    text = re.sub(r"\n{3,}", "\n\n", text)    # 3+ quebras -> 2
    return text.strip()


# ------------------ Intercom / RAG pipeline ---------------------

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
    Processa um único artigo da Intercom, aplicando o pipeline de RAG:
    parse -> categorizar -> chunk -> enriquecer -> limpar (condicional) -> embed -> montar docs
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

        # Estágio 1: Parsing básico de HTML -> texto (mantemos entidades/acentos)
        html_body = content["body"]
        parsed_text = components["text_processor"].process_html_body(html_body)
        if not parsed_text:
            print(" -> Artigo pulado: sem texto após parsing.")
            continue

        # Categorização (usa título + corpo)
        title = content.get("title", f"Artigo {article_id}")
        category = components["categorizer"].categorize_article(parsed_text, title)
        print(f" -> Categoria identificada: {category}")

        # Estágio 2: Chunking por LLM
        chunks = components["chunker"].chunk_text(parsed_text)

        # Estágio 3: Enriquecimento contextual
        enriched_chunks = components["enricher"].enrich_chunks(chunks, parsed_text)
        print(f" -> Gerados e enriquecidos {len(enriched_chunks)} chunks.")

        # Estágio 4: Limpeza condicional + Embedding(title+content)
        for i, contextualized_chunk in enumerate(enriched_chunks):
            # Decide se precisa LIMPAR
            # Regra: se já estiver claro/sem marcação, usamos minimal_normalize; senão, clean_for_embedding.
            if looks_like_html_or_markdown(contextualized_chunk):
                clean_content = clean_for_embedding(contextualized_chunk)
            else:
                clean_content = minimal_normalize(contextualized_chunk)

            if not clean_content:
                continue

            # Embeddings com concatenação de título + conteúdo (melhor sinal semântico)
            embedding_input = f"{title}\n\n{clean_content}"
            embedding = components["embedding_generator"].generate(embedding_input)
            if not embedding:
                continue

            document = {
                "title": title,
                "content": clean_content,  # conteúdo limpo e conciso
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
                    "embedding_input": "title+content"
                }
            }
            documents_for_db.append(document)

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
    """Pipeline principal para buscar e processar artigos da Intercom com foco em coleção RAG."""
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Erro de configuração: {e}")
        return

    components = {
        "intercom_client": IntercomClient(),
        "text_processor": TextProcessor(),
        "chunker": LLMChunker(),
        "enricher": ContextualEnricher(),
        "categorizer": ArticleCategorizer(),
        "embedding_generator": EmbeddingGenerator(),
        "mongodb_client": MongoDBClient()
    }

    print("🚀 Iniciando pipeline de processamento de artigos da Intercom...")

    # ID da coleção RAG (defina se quiser restringir)
    RAG_COLLECTION_ID = None  # ex.: "123456"

    # IDs a excluir (opcional)
    EXCLUDED_ARTICLE_IDS = ["7861154"]

    # (Opcional) Listar coleções
    # list_all_collections(components["intercom_client"])

    intercom_data = fetch_all_articles_from_collection(
        components["intercom_client"],
        collection_id=RAG_COLLECTION_ID
    )

    if not intercom_data or "data" not in intercom_data:
        print("⚠️ Nenhum artigo da Intercom encontrado para processar.")
        return

    print(f"📊 Total de artigos encontrados: {len(intercom_data['data'])}")

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

    print(f"\n📈 Resumo do processamento:")
    print(f" • Artigos processados: {processed_count}")
    print(f" • Artigos pulados: {skipped_count}")
    print(f" • Total de documentos gerados: {len(all_processed_documents)}")

    if all_processed_documents:
        print(f"\n💾 Salvando {len(all_processed_documents)} documentos no MongoDB...")
        components["mongodb_client"].upsert_documents(all_processed_documents)
        print("✅ Documentos salvos com sucesso!")
    else:
        print("❌ Nenhum documento foi gerado a partir dos artigos da Intercom.")

    print("\n🎉 Pipeline de artigos da Intercom concluído!")


if __name__ == "__main__":
    main()
