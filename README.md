# Intercom RAG Pipeline

Pipeline automatizado para processar artigos do Intercom Help Center usando **Contextual Retrieval** da Anthropic, com análise de imagens por VLM e chunking semântico inteligente.

## 🚀 Funcionalidades

- **Busca incremental** de artigos via API do Intercom
- **Processamento de imagens** com GPT-4 Vision para converter imagens em descrições textuais
- **LLM Chunking** semântico que usa GPT para dividir o texto de forma inteligente
- **Contextual Retrieval** baseado na metodologia da Anthropic para melhorar precisão de busca em ~35%
- **Embeddings vetoriais** com OpenAI `text-embedding-3-small`
- **Storage MongoDB** com upsert idempotente
- **Suporte multilíngue** com filtros personalizáveis por artigo
- **Categorização automática** de artigos por tipo de conteúdo

## 🏗️ Arquitetura do Pipeline

```
Intercom API → Image Processing (GPT-4V) → HTML→Markdown → Categorização → 
LLM Chunking → Contextual Enrichment → Limpeza Condicional → Embeddings → MongoDB Storage
```

### Etapas do Processamento

1. **Parsing & Image Analysis**
   - Converte HTML para Markdown preservando estrutura semântica
   - Usa GPT-4 Vision para descrever imagens em contexto
   - Remove imagens decorativas de cabeçalhos automaticamente

2. **Categorização Inteligente**
   - Classifica artigos em: `technical_support`, `features`, `billing_plans_and_pricing`, `troubleshooting`, `how_to`
   - Usa LLM para análise de título + conteúdo

3. **LLM Chunking Semântico**
   - GPT decide onde dividir o texto baseado no conteúdo
   - Separa por plataformas (Kyte PDV, Kyte Web)
   - Mantém coerência semântica em cada chunk

4. **Contextual Enrichment**
   - Adiciona contexto a cada chunk usando metodologia da Anthropic
   - Formato: `"Contexto: [explicação]\n---\n[chunk original]"`
   - Melhora significativamente a precisão do retrieval

5. **Limpeza Condicional**
   - Aplica limpeza inteligente apenas quando necessário
   - Preserva formatação semântica até o final do pipeline
   - Remove apenas ruídos visuais para otimizar embeddings

## 📁 Estrutura do Projeto

```
intercom-rag-pipeline/
├── .env                           # Variáveis de ambiente
├── .gitignore                     # Arquivos ignorados pelo Git
├── requirements.txt               # Dependências Python
├── README.md                      # Esta documentação
├── run_intercom_pipeline.py       # Script principal
├── config/
│   └── settings.py               # Configurações centralizadas
└── src/
    ├── api/
    │   └── intercom_client.py    # Cliente API Intercom
    ├── processing/
    │   ├── image_processor.py     # Análise de imagens (GPT-4V)
    │   ├── text_processor.py      # HTML→Markdown preservando estrutura
    │   ├── chunker.py            # LLM Chunking semântico
    │   ├── contextual_enricher.py # Contextual Retrieval
    │   └── categorizer.py        # Categorização automática
    ├── mongodb/
    │   └── mongodb_client.py     # Cliente MongoDB
    └── utils/
        ├── embeddings.py         # Geração de embeddings
        └── text_cleaner.py       # Limpeza condicional unificada
```

## ⚡ Instalação

### Pré-requisitos
- Python 3.8+
- MongoDB Atlas ou instância local
- Conta OpenAI com acesso à API
- Token de acesso do Intercom

### Setup Rápido

1. **Clone o repositório**
```bash
git clone <seu-repo>
cd intercom-rag-pipeline
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas credenciais:
```env
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Intercom
INTERCOM_API_TOKEN=your-intercom-token
INTERCOM_BASE_URL=https://api.intercom.io

# MongoDB
MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/
KYTE_DBNAME_AI=kyte-ai
KYTE_COLLECTION_NAME=KyteFAQKnowledgeBase

# Modelos AI
EMBEDDING_MODEL=text-embedding-3-small
RAG_SYNTH_MODEL=gpt-4o-mini
RAG_IMAGE_PROCESSOR_MODEL=gpt-4o
RAG_CONTEXTUAL_ENRICHER_MODEL=gpt-4o-mini
RAG_CHUNKER_MODEL=gpt-4o-mini
RAG_CATEGORIZER_MODEL=gpt-4o-mini

# Configurações
MAX_CHUNK_SIZE=2000
EMBEDDING_DIMENSIONS=1536
```

4. **Execute o pipeline**
```bash
python run_intercom_pipeline.py
```

## ⚙️ Configuração Avançada

### Filtros de Idioma por Artigo

O pipeline suporta configuração granular de idiomas:

```python
# IDs que devem processar todos os idiomas (PT, EN, ES)
MULTILINGUAL_ARTICLE_IDS = [
    "7861149", "7915496", "8411647", "8887223", "7915619",
    "7861109", "10008263", "7885145", "7992438", "7914908"
]

# Demais artigos processam apenas PT-BR
# Artigos excluídos do processamento
EXCLUDED_ARTICLE_IDS = ["7861154"]
```

### Filtro por Coleção RAG

```python
# Para processar apenas artigos de uma coleção específica
RAG_COLLECTION_ID = "123456"  # None para processar todos
```

## 🎯 Uso

### Execução Básica
```bash
python run_intercom_pipeline.py
```

### Logs Detalhados
O pipeline fornece logs completos do processamento:

```
🚀 Iniciando pipeline de processamento de artigos da Intercom...
📋 Pipeline: HTML → Markdown → Categorizar → Chunking → Enriquecimento → Limpeza → Embeddings
🌍 Estratégia de idiomas: PT-BR por padrão, múltiplos idiomas para artigos específicos

📊 Total de artigos encontrados: 45
📄 Processando Artigo ID: 7861149, Idioma: pt-BR, Estado: published
 -> Categoria identificada: how_to
 -> Iniciando chunking semântico...
 -> Gerados 3 chunks semânticos
 -> Iniciando enriquecimento contextual...
 -> Enriquecidos 3 chunks com contexto
✅ Artigo 7861149 processado: 3 documentos gerados

📈 Resumo do processamento:
 • Artigos processados: 42
   - Multilíngues (PT/EN/ES): 10
   - Apenas PT-BR: 32
 • Artigos pulados: 3
 • Total de documentos gerados: 156

💾 Salvando 156 documentos no MongoDB...
✅ Documentos salvos com sucesso!
```

## 📊 Estrutura dos Dados

### Documento no MongoDB
```json
{
  "title": "Como cadastrar produtos fracionados no Kyte",
  "content": "[Contexto: Este chunk explica o cadastro de produtos fracionados no Kyte PDV] Para cadastrar um produto fracionado...",
  "category": "how_to",
  "language": "pt-BR",
  "embedding": [0.123, -0.456, ...],
  "meta_data": {
    "source_type": "intercom_help_center_article",
    "article_id": "12260744",
    "intercom_url": "https://docs.kyteapp.com/...",
    "intercomCreatedAt": "2024-01-15T10:30:00Z",
    "intercomUpdatedAt": "2024-01-20T14:45:00Z",
    "article_state": "published",
    "rag_collection_id": null,
    "is_chunked": true,
    "chunk_index": 0,
    "total_chunks": 4,
    "embedding_model": "text-embedding-3-small",
    "dimensions": 1536,
    "is_multilingual_article": false
  }
}
```

### Chave de Idempotência
```javascript
{
  "meta_data.article_id": "12260744",
  "meta_data.language": "pt-BR", 
  "meta_data.chunk_index": 0
}
```

## 📚 Referências

- [Contextual Retrieval - Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- [Intercom API Documentation](https://developers.intercom.com/docs/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [MongoDB Python Driver](https://pymongo.readthedocs.io/)
- [Best Practices for RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)

---

## 🎯 Quick Start Checklist

- [ ] Clone o repositório
- [ ] Instalar dependências (`pip install -r requirements.txt`)
- [ ] Configurar `.env` com todas as chaves
- [ ] Testar conexões (OpenAI, Intercom, MongoDB)
- [ ] Executar pipeline (`python run_intercom_pipeline.py`)
- [ ] Verificar dados no MongoDB
- [ ] Configurar filtros de idioma se necessário
