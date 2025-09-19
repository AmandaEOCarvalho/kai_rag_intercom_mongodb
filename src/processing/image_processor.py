import requests
import io
import base64
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
from config.settings import Config

REFUSAL_SNIPPETS = (
    "não posso ver", "não consigo ver", "não posso analisar",
    "i can't view", "i cannot view", "unable to view", "can't see",
)

class ImageProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def _is_animated(self, image: Image.Image, response_headers: dict) -> bool:
        ct = (response_headers.get("Content-Type") or "").lower()
        if "gif" in ct:
            return True
        try:
            if getattr(image, "is_animated", False):
                return True
            n_frames = getattr(image, "n_frames", 1)
            return n_frames and n_frames > 1
        except Exception:
            return False

    def _should_skip_by_size(self, image: Image.Image) -> bool:
        # ícones / selos / logos pequenos raramente têm valor semântico para RAG
        w, h = image.size
        return (w < 80 or h < 80)

    def _sanitize_caption(self, text: str) -> str | None:
        if not text:
            return None
        t = " ".join(text.strip().split())
        # bloqueia respostas de recusa / placeholders
        low = t.lower()
        if any(snippet in low for snippet in REFUSAL_SNIPPETS):
            return None
        # força 1 sentença curta
        if len(t) > 220:
            t = t[:220].rsplit(" ", 1)[0] + "..."
        return t

    def describe_image(self, image_url: str) -> str | None:
        """
        Descreve a imagem de forma concisa. Retorna None se não valer a pena injetar.
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(image_url, headers=headers, timeout=15)
            resp.raise_for_status()

            try:
                image = Image.open(io.BytesIO(resp.content))
            except UnidentifiedImageError:
                print(f"    -> Conteúdo não é imagem, pulando")
                return None

            # Pular GIFs / animadas
            if self._is_animated(image, resp.headers):
                print(f"    -> GIF/animada detectada, pulando")
                return None

            # Pular ícones muito pequenos
            if self._should_skip_by_size(image):
                print(f"    -> Ícone pequeno ({image.size[0]}x{image.size[1]}), pulando")
                return None

            # Normaliza para JPEG opaco
            if image.mode in ('RGBA', 'P', 'LA'):
                bg = Image.new('RGB', image.size, (255, 255, 255))
                bg.paste(image.convert('RGBA'),
                         mask=image.convert('RGBA').split()[-1] if 'A' in image.getbands() else None)
                image = bg
            else:
                image = image.convert('RGB')

            # Redimensiona (mantém proporção) e comprime
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Chamada à LLM (resposta curta e direta, sem desculpas)
            payload = {
                "model": Config.RAG_IMAGE_PROCESSOR_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text":
                         "Descreva esta tela do Kyte em 1–2 frases objetivas para tutorial. "
                         "Foque em seção/ação/botões visíveis. Não inclua desculpas ou avisos."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}" }},
                    ],
                }],
                "max_tokens": 80,
                "temperature": 0.2,
            }

            completion = self.client.chat.completions.create(**payload)
            raw = completion.choices[0].message.content
            return self._sanitize_caption(raw)

        except requests.exceptions.RequestException as e:
            print(f"    ❌ Erro de rede ao processar imagem {image_url}: {e}")
            return None
        except Exception as e:
            print(f"    ❌ Erro ao processar imagem {image_url}: {e}")
            return None
