# Intercom RAG Pipeline

Pipeline automatizado para processar artigos do Intercom Help Center usando **Contextual Retrieval** da Anthropic, com análise de imagens por VLM e chunking semântico inteligente.

## Funcionalidades

- **Busca incremental** de artigos via API do Intercom
- **Processamento de imagens** com GPT-4 Vision para converter imagens em descrições textuais
- **LLM Chunking** semântico que usa GPT para dividir o texto de forma inteligente
- **Contextual Retrieval** baseado na metodologia da Anthropic para melhorar precisão de busca em ~35%
- **Embeddings vetoriais** com OpenAI `text-embedding-3-small`
- **Storage MongoDB** com upsert idempotente

## Arquitetura do Pipeline

```
1. Intercom API → 2. Image Processing (GPT-4V) → 3. LLM Chunking → 
4. Contextual Enrichment → 5. Embeddings → 6. MongoDB Storage
```

### Etapas do Processamento

1. **Parsing & Image Analysis**
   - Converte HTML para texto limpo
   - Usa GPT-4 Vision para descrever imagens em contexto
   - Remove imagens decorativas de cabeçalhos

2. **LLM Chunking Semântico**
   - GPT decide onde dividir o texto baseado no conteúdo
   - Separa por plataformas (Kyte PDV, Kyte Web)
   - Mantém coerência semântica em cada chunk

3. **Contextual Enrichment**
   - Adiciona contexto a cada chunk usando metodologia da Anthropic
   - Formato: `"Contexto: [explicação] [chunk original]"`
   - Melhora significativamente a precisão do retrieval

## Estrutura do Projeto

```
intercom-rag-pipeline/
├── .env                        # Variáveis de ambiente
├── .gitignore                  # Arquivos ignorados pelo Git
├── requirements.txt            # Dependências Python
├── README.md                   # Esta documentação
├── config/
│   └── settings.py            # Configurações centralizadas
├── src/
│   ├── api/
│   │   └── intercom_client.py # Cliente API Intercom
│   ├── processing/
│   │   ├── image_processor.py  # Análise de imagens (GPT-4V)
│   │   ├── text_processor.py   # Processamento HTML→Markdown
│   │   ├── chunker.py         # LLM Chunking semântico
│   │   └── contextual_enricher.py # Contextual Retrieval
│   ├── storage/
│   │   └── mongodb_client.py  # Cliente MongoDB
│   └── utils/
│       └── embeddings.py      # Geração de embeddings
└── main.py                    # Pipeline principal
```

## Instalação

### Pré-requisitos
- Python 3.8+
- MongoDB Atlas ou instância local
- Conta OpenAI com acesso à API
- Token de acesso do Intercom

### Setup

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
OPENAI_API_KEY=sk-your-key-here
INTERCOM_API_TOKEN=your-intercom-token
MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/
KYTE_DBNAME_AI=kyte-ai
```

4. **Execute o pipeline**
```bash
python main.py
```

## Configuração

### Variáveis de Ambiente Obrigatórias

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `OPENAI_API_KEY` | Chave da API OpenAI | `sk-proj-...` |
| `INTERCOM_API_TOKEN` | Token de acesso Intercom | `dG9rXYZ...` |
| `MONGODB_CONNECTION_STRING` | String de conexão MongoDB | `mongodb+srv://...` |

### Configurações Opcionais

| Variável | Padrão | Descrição |
|----------|---------|-----------|
| `KYTE_DBNAME_AI` | `kyte-ai` | Nome do database MongoDB |
| `MAX_CHUNK_SIZE` | `2000` | Tamanho máximo dos chunks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embedding |

## Uso

### Execução Básica
```bash
python main.py
```

### Processamento Incremental (Futuro)
```bash
# Buscar apenas artigos atualizados desde última execução
python main.py --incremental

# Processar artigo específico
python main.py --article-id 12260744
```

## Estrutura dos Dados

### Documento no MongoDB
```json
{
  "title": "Como cadastrar produtos fracionados no Kyte",
  "content": "Contexto: Este documento aborda... [chunk original]",
  "category": "help_center",
  "language": "pt-BR",
  "embedding": [0.123, -0.456, ...],
  "meta_data": {
    "source_type": "intercom_contextual",
    "article_id": "12260744",
    "intercom_url": "https://docs.kyteapp.com/...",
    "is_chunked": true,
    "chunk_index": 0,
    "total_chunks": 4,
    "embedding_model": "text-embedding-3-small",
    "dimensions": 1536
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

## Performance e Custos

### Métricas Esperadas
- **Melhoria no retrieval**: ~35% de precisão adicional (baseado no paper da Anthropic)
- **Chunking semântico**: Chunks mais coerentes vs divisão mecânica
- **Processamento de imagens**: 100% das imagens convertidas para texto

### Custos Estimados (OpenAI)
- **GPT-4V (imagens)**: ~$0.01 por imagem processada
- **GPT-4o-mini (chunking)**: ~$0.001 por artigo
- **GPT-4o-mini (contexto)**: ~$0.002 por chunk
- **Embeddings**: ~$0.0001 por chunk

## Desenvolvimento

### Adicionando Novos Processadores
```python
# src/processing/new_processor.py
class NewProcessor:
    def process(self, data):
        # Sua lógica aqui
        return processed_data
```

### Executando Testes
```bash
# Executar com apenas 1 artigo para teste
python main.py --test-mode
```

## Roadmap

### Próximas Implementações
- [ ] **Sincronização incremental** com `updated_at`
- [ ] **Scheduler automatizado** (cron/Cloud Scheduler)  
- [ ] **Webhooks Intercom** para sync em tempo real
- [ ] **Sistema de checkpoint** (`SyncMeta` collection)
- [ ] **Categorização automática** de artigos
- [ ] **Métricas e dashboards**
- [ ] **BM25 + Reranking** para 67% de melhoria (paper Anthropic)

### Melhorias Técnicas
- [ ] Cache de embeddings para evitar reprocessamento
- [ ] Batch processing para otimização de custos
- [ ] Retry logic robusto para APIs
- [ ] Logging estruturado
- [ ] Testes unitários

## Troubleshooting

### Problemas Comuns

**Erro: "OPENAI_API_KEY not found"**
```bash
# Verifique se o .env está configurado corretamente
echo $OPENAI_API_KEY
```

**MongoDB Connection Timeout**
```bash
# Verifique se o IP está liberado no MongoDB Atlas
# Teste a conexão manualmente
```

**Rate Limit OpenAI**
```bash
# O pipeline tem retry automático, mas pode ser necessário
# reduzir o volume de processamento simultâneo
```

## Contribuição

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Referências

- [Contextual Retrieval - Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- [Intercom API Documentation](https://developers.intercom.com/docs/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)