import sys
import os
import re
import unicodedata  # >>> para normalização robusta

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

# -------------------- Limpeza “inteligente” p/ embeddings --------------------

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
MD_HEADING_MARK_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_SINGLE_ASTERISK_LINE_RE = re.compile(r"(?m)^\s*\*\s*$")
MD_HR_RE = re.compile(r"(?m)^\s{0,3}[-*_]{3,}\s*$")
MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
MD_ITALIC_RE = re.compile(r"\*([^*]+)\*|_([^_]+)_")
_HRULE_RE = re.compile(r"(?m)^\s*(?:\*[\s*]{2,}|[-_]{3,})\s*$")   # *** , * * * , --- , ___
EXTRANEOUS_TOKENS_RE = re.compile(r"(?:\u200b|\u200c|\u200d|\uFEFF)")  # zero-width etc.

def looks_like_html_or_markdown(text: str) -> bool:
    """Heurística mais robusta para decidir quando LIMPAR."""
    if HTML_TAGS_RE.search(text):
        return True
    if "```" in text or "`" in text:
        return True
    if "**" in text or "__" in text:
        return True
    if MD_HR_RE.search(text):
        return True
    if re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", text):  # headings #..######
        return True
    if re.search(r"\[[^\]]+\]\([^)]+\)", text):      # [texto](url)
        return True
    if re.search(r"(?m)^\s*[-*•]\s+\S", text):       # bullets
        return True
    return False

def minimal_normalize(text: str) -> str:
    """Quando NÃO limpar: só normaliza espaços/quebras sem perder semântica."""
    text = _CONTROL_CHARS_RE.sub("", text)
    text = EXTRANEOUS_TOKENS_RE.sub("", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)          # colapsa espaços
    text = re.sub(r"\n{3,}", "\n\n", text)           # 3+ quebras -> 2
    return text.strip()

def strip_markdown_and_html(text: str) -> str:
    """Remove HTML/Markdown mantendo o conteúdo semântico."""
    # HTML
    text = HTML_TAGS_RE.sub(" ", text)
    # Markdown
    text = MD_CODE_FENCE_RE.sub(" ", text)           # ```code``` -> remove
    text = MD_INLINE_CODE_RE.sub(r"\1", text)        # `inline` -> inline
    text = MD_HEADING_MARK_RE.sub("", text)          # remove marcadores de heading
    text = MD_HR_RE.sub(" ", text)                   # ---- -> espaço
    text = _SINGLE_ASTERISK_LINE_RE.sub("", text)
    text = _HRULE_RE.sub(" ", text)
    text = MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    # bullets: normaliza para “- ”
    text = re.sub(r"(?m)^\s*[\*\•]\s+", "- ", text)
    return text

# ----- Remoção robusta dos headings tipo “What you’ll learn / O que você vai aprender / Lo que vas a aprender” -----

_LEARN_CANONICAL = {
    "what youll learn",
    "what you will learn",
    "o que voce vai aprender",
    "lo que vas a aprender",
}

_HR_LINE_RE = re.compile(r"(?m)^\s*(?:[-_*]\s*){3,}\s*$")

def _ascii_fold(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("’", "'").replace("“", '"').replace("”", '"')

def _strip_md_heading_markup(line: str) -> str:
    line = re.sub(r"^\s{0,3}(?:\#{1,6}\s*)?", "", line)   # #, ##, ...
    line = re.sub(r"^\s*(?:\*\*|__)\s*", "", line)        # ** no início
    line = re.sub(r"\s*(?:\*\*|__)\s*$", "", line)        # ** no fim
    return line.strip()

def _normalize_line_for_compare(line: str) -> str:
    line = _strip_md_heading_markup(line)
    line = _ascii_fold(line).lower()
    line = line.replace("'", "")                # you'll -> youll
    line = re.sub(r"\s+", " ", line)
    return line.strip(" :.-")

def drop_learning_leadins(text: str) -> str:
    """Remove a linha do heading + linhas decorativas em volta, se existirem."""
    lines = text.splitlines()
    keep = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        norm = _normalize_line_for_compare(raw)
        if norm in _LEARN_CANONICAL:
            # se a linha anterior for régua/vazia, remove também
            if keep and (_HR_LINE_RE.match(keep[-1]) or keep[-1].strip() == ""):
                keep.pop()
            # pula a linha do heading
            i += 1
            # pula linhas vazias/régua subsequentes
            while i < len(lines) and (lines[i].strip() == "" or _HR_LINE_RE.match(lines[i])):
                i += 1
            continue
        keep.append(raw)
        i += 1
    return "\n".join(keep)

def clean_for_embedding(text: str) -> str:
    """
    LIMPA quando necessário:
      - remove HTML/Markdown/artefatos
      - remove emojis e caracteres de controle
      - normaliza espaços/quebras
    Mantém acentos, 'R$', '%', e-mails, @, etc.
    """
    text = _EMOJI_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = EXTRANEOUS_TOKENS_RE.sub("", text)

    if looks_like_html_or_markdown(text):
        text = strip_markdown_and_html(text)

    text = re.sub(r"[^\S\r\n]+", " ", text)   # colapsa espaços
    text = re.sub(r"\n{3,}", "\n\n", text)    # 3+ quebras -> 2
    return text.strip()

def clean_context_then_text(content: str) -> str:
    """
    Compatível com seu formato de teste:
    - Se começar com 'Contexto:' até '---', mantém como [Contexto: ...] antes do texto.
    - Depois aplica a limpeza inteligente (condicional).
    """
    # Isola contexto (opcional)
    context_match = re.match(r'^(Contexto:.*?)\n---\n(.*)', content, re.DOTALL)
    if context_match:
        context_part = context_match.group(1).strip()
        main_content = context_match.group(2).strip()
        base = f"[{context_part}] {main_content}"
    else:
        base = content

    # >>> remove “What you’ll learn / O que você vai aprender / Lo que vas a aprender”
    base = drop_learning_leadins(base)

    # Decide “quando limpar” vs “quando não limpar”
    if looks_like_html_or_markdown(base):
        return clean_for_embedding(base)
    return minimal_normalize(base)

# -------------------- Regras de elegibilidade --------------------

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

# -------------------- Pipeline de TESTE (sem Mongo) --------------------

def process_single_article_test(article: dict, components: dict, rag_collection_id: str = None, excluded_article_ids: list = None) -> list:
    """Versão de teste que processa um artigo mas NÃO salva no MongoDB."""
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

        print(f"\n📄 TESTANDO Artigo ID: {article_id}, Idioma: {lang}, Estado: {state}")
        title = content.get("title", "Sem título")
        print(f"📝 Título: {title}")

        # Estágio 1: Parsing de HTML -> texto (mantém entidades/acentos)
        parsed_text = components["text_processor"].process_html_body(content["body"])
        if not parsed_text:
            print(" -> Artigo pulado pois não contém texto após o parsing.")
            continue

        print(f"📏 Texto parseado: {len(parsed_text)} chars")
        print(f"🔤 Prévia (200): {parsed_text[:200]}...\n")

        # Categorização
        category = components["categorizer"].categorize_article(parsed_text, title)
        print(f"🏷️  Categoria identificada: {category}")

        # Estágio 2: Chunking com LLM
        print("✂️  Iniciando chunking...")
        chunks = components["chunker"].chunk_text(parsed_text)
        print(f"✂️  Gerados {len(chunks)} chunks iniciais")

        # Estágio 3: Enriquecimento Contextual
        print("🔧 Iniciando enriquecimento contextual...")
        enriched_chunks = components["enricher"].enrich_chunks(chunks, parsed_text)
        print(f"🔧 Gerados e enriquecidos {len(enriched_chunks)} chunks finais")

        # Mostra exemplo de chunk enriquecido bruto
        if enriched_chunks:
            print("\n📋 EXEMPLO DE CHUNK ENRIQUECIDO (bruto):")
            print(f"Tamanho: {len(enriched_chunks[0])} chars")
            print(f"Conteúdo (300): {enriched_chunks[0][:300]}...\n")

        # Estágio 4: Limpeza condicional + Embedding(title+content)
        for i, contextualized_chunk in enumerate(enriched_chunks):
            print(f"🧽 Limpando chunk {i+1} ...")
            clean_content = clean_context_then_text(contextualized_chunk)
            if not clean_content:
                print(f"   ⚠️ Chunk {i+1} vazio após limpeza, pulando.")
                continue

            embedding_input = f"{title}\n\n{clean_content}"

            # Prévia para inspeção
            print(f"📦 EMBEDDING INPUT (chunk {i+1}) — len={len(embedding_input)}:")
            print(embedding_input[:400] + ("..." if len(embedding_input) > 400 else ""))
            print()

            # Gerar embedding (usa seu gerador atual)
            print(f"🧠 Gerando embedding para chunk {i+1}...")
            embedding = components["embedding_generator"].generate(embedding_input)
            if not embedding:
                print(f"   ❌ Falha ao gerar embedding para chunk {i+1}")
                continue

            print(f"   ✅ Embedding gerado: {len(embedding)} dimensões")

            # Imprimir CONTEÚDO FINAL LIMPO do chunk (completo)
            print(f"\n📋 CONTEÚDO FINAL LIMPO DO CHUNK {i+1} (len={len(clean_content)}):")
            print(clean_content)
            print("-" * 70)

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
                    "embedding_input": "title+content"
                }
            }
            documents_for_db.append(document)

        print(f"✅ Artigo {article_id} processado com sucesso!")
        print("=" * 80)

    return documents_for_db

def main():
    """Versão de teste para processar apenas 3 artigos sem salvar no MongoDB."""
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Erro de configuração: {e}")
        return

    # Inicializar componentes (SEM MongoDB)
    components = {
        "intercom_client": IntercomClient(),
        "text_processor": TextProcessor(),
        "chunker": LLMChunker(),
        "enricher": ContextualEnricher(),
        "categorizer": ArticleCategorizer(),
        "embedding_generator": EmbeddingGenerator()
    }

    print("🧪 MODO TESTE - Processando apenas 3 artigos (SEM salvar no MongoDB)")
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
        processed_docs = process_single_article_test(article, components, RAG_COLLECTION_ID, EXCLUDED_ARTICLE_IDS)
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
        print(f"   Metadados: {list(example_doc['meta_data'].keys())}")

    print("\n🎉 Teste concluído! Nada foi salvo no banco de dados.")

if __name__ == "__main__":
    main()
