import re
from openai import OpenAI
from config.settings import Config

class LLMChunker:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.max_chunk_size = Config.MAX_CHUNK_SIZE
    
    def chunk_text(self, full_text: str) -> list:
        """Implementa LLM Chunking semântico"""
        # Divisão preliminar
        # Divide o texto sempre que encontrar um cabeçalho de qualquer nível (#, ##, ###).
        preliminary_chunks = re.split(r'\n(?=#+ )', full_text)
        preliminary_chunks = [chunk.strip() for chunk in preliminary_chunks if chunk and chunk.strip()]

        if len(preliminary_chunks) <= 1:
            print("  -> Apenas um chunk. Retornando sem LLM.")
            return preliminary_chunks

        # Formatar para o LLM
        formatted_for_llm = ""
        for i, chunk in enumerate(preliminary_chunks):
            formatted_for_llm += f"start {i}\n{chunk}\nend {i}\n\n"

        prompt = f"""
        Sua tarefa é analisar o texto de um tutorial e identificar os melhores pontos para dividi-lo em chunks semanticamente lógicos e autocontidos.
        O texto foi pré-dividido em seções marcadas com "start X" e "end X".
        Avalie o conteúdo e decida onde as quebras fazem mais sentido com base nos seguintes princípios:

        1.  **Separar Plataformas:** Tutoriais para plataformas (ex: 'Kyte PDV', 'Kyte Web') ou planos (ex: 'PRO', 'GROW') diferentes devem estar em chunks separados.
        2.  **Isolar Seções Distintas:** Seções introdutórias (como 'O que você vai aprender'), guias passo a passo ('Passo a passo') e seções de dicas ('Dicas úteis') são bons candidatos a chunks independentes se forem suficientemente longos ou distintos.
        3.  **Manter a Coerência:** O objetivo é que cada chunk cubra um tópico ou uma tarefa completa. Evite dividir listas de passos numerados no meio.

        Com base na sua análise, responda APENAS com os números dos chunks APÓS os quais uma nova divisão principal deve ocorrer, separados por vírgula.

        Texto para análise:
        {formatted_for_llm}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=Config,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            split_suggestions = response.choices[0].message.content.strip()
            split_indices = {int(i) for i in re.findall(r'\d+', split_suggestions)}
        except Exception as e:
            print(f"Erro no LLM Chunking: {e}")
            return preliminary_chunks

        # Reagrupar chunks
        final_chunks = []
        current_chunk_group = []
        for i, chunk in enumerate(preliminary_chunks):
            current_chunk_group.append(chunk)
            if i in split_indices:
                final_chunks.append("\n\n".join(current_chunk_group))
                current_chunk_group = []
        
        if current_chunk_group:
            final_chunks.append("\n\n".join(current_chunk_group))
            
        return final_chunks