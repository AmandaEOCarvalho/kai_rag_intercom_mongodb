import re
from bs4 import BeautifulSoup
import html2text
from .image_processor import ImageProcessor

class TextProcessor:
    """
    Processa HTML em Markdown formatado, preservando estrutura semântica.
    Remove apenas ruídos visuais, mantendo formatação útil para chunking e contextual enrichment.
    """
    
    def __init__(self):
        self.image_processor = ImageProcessor()

    def _clean_visual_noise_only(self, md: str) -> str:
        """
        Remove APENAS ruídos visuais, preservando formatação semântica.
        Esta é a diferença chave: não remove headings, bold, listas, etc.
        """
        # Remove linhas decorativas (réguas visuais como *** --- ___)
        md = re.sub(r'^\s*(\* ?\* ?\*|\*{3,}|-{3,}|_{3,})\s*$', '', md, flags=re.MULTILINE)
        
        # Remove emojis (ruído visual para tutoriais técnicos)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # símbolos & pictogramas
            "\U0001F680-\U0001F6FF"  # transportes & mapas
            "\U0001F1E0-\U0001F1FF"  # bandeiras
            "\U00002500-\U00002BEF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", re.UNICODE
        )
        md = emoji_pattern.sub('', md)
        
        # Compacta quebras excessivas (3+ quebras → 2)
        md = re.sub(r'\n{3,}', '\n\n', md)
        
        # ✅ PRESERVA intencionalmente:
        # - Headings (# ## ### etc.)
        # - Bold (**texto** __texto__)
        # - Italic (*texto* _texto_)
        # - Links [texto](url)
        # - Listas (- * 1.)
        # - Code blocks (```)
        # - Inline code (`código`)
        
        return md

    def _maybe_use_alt(self, img_tag) -> str | None:
        """Usa alt text se for descritivo (6-160 chars)"""
        alt = (img_tag.get('alt') or '').strip()
        if 6 <= len(alt) <= 160:
            return " ".join(alt.split())
        return None

    def process_html_body(self, html_body: str) -> str:
        """
        Converte HTML para Markdown formatado preservando estrutura semântica.
        
        Pipeline:
        1. Substitui imagens por descrição concisa (ou remove)
        2. Converte HTML em Markdown PRESERVANDO formatação
        3. Remove apenas ruídos visuais (não semânticos)
        
        Args:
            html_body (str): HTML do corpo do artigo
            
        Returns:
            str: Markdown formatado pronto para chunking e contextual enrichment
        """
        if not html_body:
            return ""

        soup = BeautifulSoup(html_body, 'html.parser')

        # Etapa 1: Processa imagens
        for img in soup.find_all('img'):
            url = img.get('src')
            if not url:
                img.decompose()
                continue

            print(f"  -> Processando imagem: {url[:50]}...")

            # Imagens dentro de headings são decorativas
            if img.find_parent(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                img.decompose()
                continue

            # Usa alt text curto quando disponível; senão GPT-4o
            desc = self._maybe_use_alt(img) or self.image_processor.describe_image(url)

            if not desc:
                print(f"    -> Imagem removida (sem descrição útil)")
                img.decompose()
                continue

            # Injeta descrição inline, sem quebras extras
            replacement = soup.new_string(f"[Descrição da Imagem: {desc}]")
            img.replace_with(replacement)

        # Etapa 2: Converte HTML para Markdown PRESERVANDO formatação
        h = html2text.HTML2Text()
        h.body_width = 0           # Sem quebras forçadas de linha
        h.protect_links = True     # ✅ Preserva links intactos
        h.wrap_links = False       # ✅ Não quebra links longos
        h.unicode_snob = True      # ✅ Preserva caracteres Unicode (acentos)
        h.escape_snob = True       # ✅ Evita escaping desnecessário
        
        md = h.handle(str(soup))

        # Etapa 3: Remove APENAS ruídos visuais (preserva formatação semântica)
        md = self._clean_visual_noise_only(md)

        return md.strip()