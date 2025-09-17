from config.settings import Config
from src.utils.embeddings import EmbeddingGenerator
from src.mongodb.mongodb_client import MongoDBClient
import json
import re
from pathlib import Path
from typing import Dict, Any, List

class SemanticChunker:
    """Classe para fazer chunking semântico inteligente de textos"""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """
        Faz chunking semântico do texto priorizando quebras naturais
        
        Prioridade de quebras:
        1. Linhas duplas ou triplas (\n\n+ ou \n\n\n+)
        2. Pontos seguidos de maiúscula
        3. Quebras forçadas se muito grande
        """
        chunks = []
        
        if not text.strip():
            return chunks
        
        # Primeiro, tentar quebrar por linhas duplas/triplas
        sections = re.split(r'\n\s*\n+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Se adicionar esta seção não ultrapassar o limite, adiciona
            if len(current_chunk + "\n\n" + section) <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
            else:
                # Finaliza chunk atual se tem conteúdo suficiente
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, title, chunk_index))
                    chunk_index += 1
                    current_chunk = section
                else:
                    # Se chunk atual é muito pequeno, força junção
                    current_chunk += "\n\n" + section
                
                # Se a seção atual é muito grande, quebra por pontos
                if len(current_chunk) > self.max_chunk_size:
                    sub_chunks = self._split_by_sentences(current_chunk)
                    for i, sub_chunk in enumerate(sub_chunks):
                        if sub_chunk.strip():
                            chunks.append(self._create_chunk(sub_chunk, title, chunk_index))
                            chunk_index += 1
                    current_chunk = ""
        
        # Adicionar último chunk se houver
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, title, chunk_index))
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Quebra texto por pontos quando necessário"""
        sentences = re.split(r'\.(?=\s+[A-Z])', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Recolocar o ponto se não terminar com pontuação
            if not sentence.endswith(('.', '!', '?', ':')):
                sentence += '.'
            
            if len(current_chunk + " " + sentence) <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, title: str, index: int) -> Dict[str, Any]:
        """Cria estrutura de chunk"""
        return {
            "content": content.strip(),
            "title": title,
            "chunk_index": index,
            "chunk_size": len(content.strip())
        }

def generate_documents_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """
    Gera documentos a partir do arquivo JSON com chunking semântico.
    
    Args:
        json_file_path: Caminho para o arquivo JSON
        
    Returns:
        Lista de documentos processados para inserção no MongoDB
    """
    documents = []
    chunker = SemanticChunker(max_chunk_size=1000, min_chunk_size=100)
    
    try:
        # Carregar o arquivo JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Suporta dict com chave 'articles', lista de dicts, ou lista de dicts com chave 'articles'
        if isinstance(data, dict) and "articles" in data:
            articles = data["articles"]
        elif isinstance(data, list):
            # Se for lista de dicts com chave 'articles', pega todos
            if len(data) > 0 and isinstance(data[0], dict) and "articles" in data[0]:
                articles = data[0]["articles"]
            else:
                articles = data
        else:
            articles = []
        print(f"📊 Encontrados {len(articles)} artigos no JSON para processar")

        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                print(f"⚠️  Pulando artigo {i+1}: formato inválido")
                continue
            if not article.get('title') and not article.get('content'):
                print(f"⚠️  Pulando artigo {i+1}: sem título nem conteúdo")
                continue
            title = article.get('title', f'Artigo {i+1}')
            content = article.get('content', '')
            print(f"📝 Processando artigo {i+1}/{len(articles)}: {title[:50]}...")
            # Verificar se o conteúdo precisa de chunking
            if len(content) <= 1000:
                # Conteúdo pequeno - criar documento único
                document = {
                    "title": title,
                    "content": content,
                    "category": article.get("category", ""),
                    "language": article.get("language", "pt-BR"),
                    "meta_data": {
                        "source_type": "json_article",
                        "source_file": Path(json_file_path).name,
                        "article_index": i,
                        "is_chunked": False,
                        "total_chunks": 1,
                        "chunk_index": 0
                    }
                }
                documents.append(document)
            else:
                # Conteúdo grande - fazer chunking
                chunks = chunker.chunk_text(content, title)
                print(f"🔪 Artigo dividido em {len(chunks)} chunks semânticos")
                for chunk_data in chunks:
                    document = {
                        "title": title,
                        "content": chunk_data["content"],
                        "category": article.get("category", ""),
                        "language": article.get("language", "pt-BR"),
                        "meta_data": {
                            "source_type": "json_article",
                            "source_file": Path(json_file_path).name,
                            "article_index": i,
                            "is_chunked": True,
                            "chunk_index": chunk_data["chunk_index"],
                            "chunk_size": chunk_data["chunk_size"],
                            "total_chunks": len(chunks)
                        }
                    }
                    documents.append(document)

        print(f"✅ Processados {len(documents)} documentos do JSON")
        return documents
        
    except FileNotFoundError:
        print(f"❌ Arquivo JSON não encontrado: {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao decodificar JSON: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao processar JSON: {e}")
        return []

def update_json_knowledge(json_file_path: str):
    """
    Pipeline dedicado a buscar, processar e salvar os documentos
    do arquivo JSON com chunking e embeddings.
    
    Args:
        json_file_path: Caminho para o arquivo JSON
    """
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Erro de configuração: {e}")
        return

    # Inicializar os componentes necessários
    embedding_generator = EmbeddingGenerator()
    mongodb_client = MongoDBClient()

    print("🚀 Iniciando pipeline de atualização de conhecimento JSON...")
    print(f"📄 Arquivo: {json_file_path}")
    
    # ETAPA 1: Gerar documentos a partir do JSON
    json_documents = generate_documents_from_json(json_file_path)
    
    if not json_documents:
        print("❌ Nenhum documento foi gerado do JSON.")
        return
    
    # ETAPA 2: Gerar embeddings para cada documento
    print(f"\n🤖 Gerando embeddings para {len(json_documents)} documentos...")
    processed_documents = []
    
    for i, doc in enumerate(json_documents):
        print(f"🔄 Gerando embedding {i+1}/{len(json_documents)}")
        
        # Texto para embedding (título + conteúdo)
        embedding_text = f"{doc['title']}. {doc['content']}"
        embedding = embedding_generator.generate(embedding_text)
        
        if embedding:
            doc['embedding'] = embedding
            doc['meta_data']['embedding_model'] = Config.EMBEDDING_MODEL
            doc['meta_data']['dimensions'] = Config.EMBEDDING_DIMENSIONS
            processed_documents.append(doc)
        else:
            print(f"❌ Erro ao gerar embedding para documento {i+1}")
    
    # ETAPA 3: Salvar os documentos no MongoDB
    if processed_documents:
        print(f"\n💾 Salvando {len(processed_documents)} documentos no MongoDB...")
        mongodb_client.upsert_documents(processed_documents)
        
        # Estatísticas
        chunked_docs = len([d for d in processed_documents if d["meta_data"]["is_chunked"]])
        single_docs = len([d for d in processed_documents if not d["meta_data"]["is_chunked"]])
        
        print(f"📊 Estatísticas:")
        print(f"   - Documentos únicos: {single_docs}")
        print(f"   - Chunks de documentos grandes: {chunked_docs}")
        print(f"   - Total processado: {len(processed_documents)}")
    else:
        print("❌ Nenhum documento foi processado com sucesso.")

    print("\n🎉 Pipeline de conhecimento JSON concluído!")

def main():
    """
    Função principal que pode ser usada para testar o pipeline
    """
    import sys
    
    if len(sys.argv) < 2:
        print("❌ Uso: python run_json_pipeline.py <caminho_para_arquivo_json>")
        print("📄 Exemplo: python run_json_pipeline.py data/artigos.json")
        return
    
    json_file_path = sys.argv[1]
    update_json_knowledge(json_file_path)

if __name__ == "__main__":
    main()