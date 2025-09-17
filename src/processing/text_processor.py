import re
from bs4 import BeautifulSoup
import html2text
from .image_processor import ImageProcessor

class TextProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()

    def _clean_markdown_noise(self, md: str) -> str:
        # remove linhas de <hr> convertidas em *** e variações
        md = re.sub(r'^\s*(\* ?\* ?\*|\*{3,}|-{3,}|_{3,})\s*$', '', md, flags=re.MULTILINE)
        # remove hashes de cabeçalho deixando só o texto
        md = re.sub(r'^\s{0,3}#{1,6}\s*', '', md, flags=re.MULTILINE)
        # compacta linhas em branco excessivas
        md = re.sub(r'\n{3,}', '\n\n', md)
        return md

    def _maybe_use_alt(self, img_tag) -> str | None:
        alt = (img_tag.get('alt') or '').strip()
        if 6 <= len(alt) <= 160:
            return " ".join(alt.split())
        return None

    def process_html_body(self, html_body: str) -> str:
        """
        1) Substitui imagens por descrição concisa (ou remove)
        2) Converte o HTML em markdown simples
        3) Remove ruídos como *** e hashes de títulos
        """
        if not html_body:
            return ""

        soup = BeautifulSoup(html_body, 'html.parser')

        for img in soup.find_all('img'):
            url = img.get('src')
            if not url:
                img.decompose()
                continue

            print(f"  -> Processando imagem: {url[:50]}...")

            # imagens dentro de headings são decorativas
            if img.find_parent(['h1','h2','h3','h4']):
                img.decompose()
                continue

            # usa alt curto quando disponível; senão GPT
            desc = self._maybe_use_alt(img) or self.image_processor.describe_image(url)

            if not desc:
                print(f"    -> Imagem removida (sem descrição útil)")
                img.decompose()
                continue

            # injeta inline, sem quebras extras
            replacement = soup.new_string(f"[Descrição da Imagem: {desc}]")
            img.replace_with(replacement)

        h = html2text.HTML2Text()
        h.body_width = 0  # sem quebras duras
        md = h.handle(str(soup))

        md = self._clean_markdown_noise(md)

        # remove emojis / pictogramas mantendo acentos e símbolos úteis
        emoji_pattern = re.compile(
            "["                       # blocos unicode com emojis/sinais
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
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

        return md.strip()
