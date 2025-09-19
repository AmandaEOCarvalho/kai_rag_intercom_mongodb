from openai import OpenAI
from config.settings import Config

class ContextualEnricher:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def enrich_chunks(self, chunks: list, full_document_text: str, language: str) -> list:
        """Adiciona contexto a cada chunk usando a metodologia "Contextual Retrieval" da Anthropic.
        O objetivo é tornar cada chunk mais autocontido para melhorar a busca (RAG).
        """
        enriched_chunks = []
        print(f"  -> Enriquecendo {len(chunks)} chunks no idioma: {language} ...")
        
        for i, chunk in enumerate(chunks):
            prompt = f"""
            Você receberá um documento completo e um chunk específico desse documento.
            Sua tarefa é gerar um contexto curto e sucinto para situar este chunk dentro do documento, com o objetivo de melhorar a recuperação da busca (search retrieval).
            O contexto deve responder a perguntas como: Qual é o tópico principal do documento? Qual subtópico ou plataforma específica (ex: Kyte PDV, Kyte Web) este chunk aborda? Qual é a intenção deste chunk (ex: passo a passo, dica, introdução)?

            <document>
            {full_document_text}
            </document>

            <chunk>
            {chunk}
            </chunk>

            Responda APENAS com o contexto sucinto e nada mais. 
            Limite sua resposta a no máximo 80 tokens.
            Sua resposta deve ser no idioma: {language}.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=Config.RAG_CONTEXTUAL_ENRICHER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=80
                )
                context = response.choices[0].message.content.strip()
                # Prepara o chunk final com o contexto
                final_chunk = f"Contexto: {context}\n\n---\n\n{chunk}"
                enriched_chunks.append(final_chunk)
                
            except Exception as e:
                # Adiciona o chunk original em caso de erro
                print(f"Erro ao enriquecer chunk {i+1}, adicionando sem contexto: {e}")
                enriched_chunks.append(chunk)

        return enriched_chunks