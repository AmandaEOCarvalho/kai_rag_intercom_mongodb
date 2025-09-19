from openai import OpenAI
from config.settings import Config

class ArticleCategorizer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.categories = [
            'technical_support', 
            'features', 
            'billing_plans_and_pricing', 
            'troubleshooting', 
            'how_to'
        ]
    
    def categorize_article(self, full_text: str, title: str) -> str:
        """Classifica um artigo em uma das categorias predefinidas"""
        prompt = f"""
        Você é um especialista em classificação de conteúdo para uma central de ajuda de software.
        Sua tarefa é analisar o título e o conteúdo de um artigo e classificá-lo em UMA das seguintes categorias:
        {', '.join(self.categories)}

        - 'how_to': Para tutoriais e guias passo a passo sobre como usar uma funcionalidade.
        - 'features': Para descrições de funcionalidades, o que são e para que servem.
        - 'troubleshooting': Para artigos que ajudam a resolver problemas, erros ou comportamentos inesperados.
        - 'billing_plans_and_pricing': Para artigos sobre preços, planos, assinaturas e cobranças.
        - 'technical_support': Para informações gerais de suporte, como comunicados, avisos de manutenção ou como entrar em contato.

        TÍTULO DO ARTIGO: "{title}"
        CONTEÚDO DO ARTIGO:
        ---
        {full_text}
        ---

        Responda APENAS com o nome da categoria mais apropriada da lista.
        """
        
        print("  -> Categorizando artigo com LLM...")
        try:
            response = self.client.chat.completions.create(
                model=Config.RAG_CATEGORIZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            category = response.choices[0].message.content.strip()
            # Garante que a resposta seja uma das categorias válidas
            if category in self.categories:
                return category
            else:
                print(f"  -> Categoria '{category}' inválida, usando 'technical_support'")
                return 'technical_support'
                
        except Exception as e:
            # Fallback em caso de erro
            print(f"Erro ao categorizar artigo: {e}")
            return 'technical_support'