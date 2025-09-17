import re
import unicodedata

class TextCleaner:
    """
    Módulo unificado para limpeza de texto seguindo as melhores práticas:
    - Limpeza condicional: só quando necessário
    - Preserva semântica (acentos, símbolos monetários, emails, etc.)
    - Remove apenas ruídos visuais para embeddings
    """
    
    def __init__(self):
        # Padrões de emojis e pictogramas
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # símbolos & pictogramas
            "\U0001F680-\U0001F6FF"  # transportes & mapas
            "\U0001F1E0-\U0001F1FF"  # bandeiras
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
            "]+", flags=re.UNICODE
        )
        
        # Caracteres de controle problemáticos
        self.control_chars_pattern = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
        
        # Tags HTML
        self.html_tags_pattern = re.compile(r"<[^>]+>")
        
        # Tokens invisíveis
        self.invisible_tokens_pattern = re.compile(r"(?:\u200b|\u200c|\u200d|\uFEFF)")
        
        # Padrões para detecção de markdown/HTML
        self.markdown_indicators = [
            re.compile(r"```.*?```", re.DOTALL),  # code blocks
            re.compile(r"`[^`]+`"),                # inline code
            re.compile(r"\*\*[^*]+\*\*"),         # bold
            re.compile(r"__[^_]+__"),             # bold alt
            re.compile(r"(?m)^\s{0,3}#{1,6}\s"),  # headings
            re.compile(r"\[[^\]]+\]\([^)]+\)"),   # links
            re.compile(r"(?m)^\s{0,3}[-*_]{3,}"), # horizontal rules
        ]
    
    def looks_like_markdown_or_html(self, text: str) -> bool:
        """
        Detecta se o texto contém marcações que precisam de limpeza estrutural.
        
        Args:
            text (str): Texto para análise
            
        Returns:
            bool: True se precisar de limpeza, False para normalização mínima
        """
        # HTML tags
        if self.html_tags_pattern.search(text):
            return True
            
        # Indicadores de markdown
        for pattern in self.markdown_indicators:
            if pattern.search(text):
                return True
                
        return False
    
    def minimal_normalize(self, text: str) -> str:
        """
        Normalização mínima: remove apenas caracteres problemáticos.
        Preserva toda semântica (acentos, R$, %, emails, etc.).
        
        Args:
            text (str): Texto para normalização mínima
            
        Returns:
            str: Texto normalizado preservando semântica
        """
        # Remove caracteres de controle
        text = self.control_chars_pattern.sub("", text)
        
        # Remove tokens invisíveis
        text = self.invisible_tokens_pattern.sub("", text)
        
        # Normaliza espaços (preserva quebras semânticas)
        text = re.sub(r"[^\S\r\n]+", " ", text)  # múltiplos espaços → 1
        text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ quebras → 2
        
        return text.strip()
    
    def clean_for_embeddings(self, text: str) -> str:
        """
        Limpeza completa para otimizar embeddings vetoriais.
        Remove marcações visuais mas preserva conteúdo semântico.
        
        Args:
            text (str): Texto com possível marcação HTML/Markdown
            
        Returns:
            str: Texto limpo otimizado para embeddings
        """
        # Remove emojis (ruído visual)
        text = self.emoji_pattern.sub("", text)
        
        # Remove caracteres de controle
        text = self.control_chars_pattern.sub("", text)
        
        # Remove tokens invisíveis
        text = self.invisible_tokens_pattern.sub("", text)
        
        # Remove HTML tags
        text = self.html_tags_pattern.sub(" ", text)
        
        # Remove estrutura markdown preservando conteúdo
        text = self._strip_markdown_structure(text)
        
        # Remove headings tipo "What you'll learn"
        text = self._remove_learning_headings(text)
        
        # Normaliza espaços
        text = re.sub(r"[^\S\r\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()
    
    def _strip_markdown_structure(self, text: str) -> str:
        """Remove estrutura markdown preservando conteúdo semântico."""
        # Code blocks → remove completamente
        text = re.sub(r"```.+?```", " ", text, flags=re.DOTALL)
        
        # Inline code → preserva conteúdo
        text = re.sub(r"`([^`]+)`", r"\1", text)
        
        # Headers → remove marcadores #
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
        
        # Bold/italic → preserva conteúdo
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        
        # Links [texto](url) → preserva texto
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        
        # Horizontal rules → remove
        text = re.sub(r"(?m)^\s{0,3}[-*_]{3,}\s*$", " ", text)
        text = re.sub(r"(?m)^\s*\*\s*$", "", text)  # linha com só *
        
        # Bullets → normaliza para -
        text = re.sub(r"(?m)^\s*[\*\•]\s+", "- ", text)
        
        return text
    
    def _ascii_fold(self, s: str) -> str:
        """Normalização Unicode para comparação"""
        s = unicodedata.normalize('NFKD', s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.replace("'", "'").replace(""", '"').replace(""", '"')
    
    def _normalize_line_for_compare(self, line: str) -> str:
        """Normaliza linha para comparação de headings"""
        # Remove marcadores de heading
        line = re.sub(r"^\s{0,3}(?:\#{1,6}\s*)?", "", line)
        line = re.sub(r"^\s*(?:\*\*|__)\s*", "", line)
        line = re.sub(r"\s*(?:\*\*|__)\s*$", "", line)
        
        # Normaliza para comparação
        line = self._ascii_fold(line).lower()
        line = line.replace("'", "")  # you'll → youll
        line = re.sub(r"\s+", " ", line)
        return line.strip(" :.-")
    
    def _remove_learning_headings(self, text: str) -> str:
        """
        Remove headings do tipo "What you'll learn / O que você vai aprender"
        """
        learning_phrases = {
            "what youll learn",
            "what you will learn", 
            "o que voce vai aprender",
            "lo que vas a aprender",
        }
        
        hr_pattern = re.compile(r"(?m)^\s*(?:[-_*]\s*){3,}\s*$")
        
        lines = text.splitlines()
        keep = []
        i = 0
        
        while i < len(lines):
            raw = lines[i]
            norm = self._normalize_line_for_compare(raw)
            
            if norm in learning_phrases:
                # Remove linha anterior se for régua/vazia
                if keep and (hr_pattern.match(keep[-1]) or keep[-1].strip() == ""):
                    keep.pop()
                
                # Pula a linha do heading
                i += 1
                
                # Pula linhas vazias/régua subsequentes
                while i < len(lines) and (lines[i].strip() == "" or hr_pattern.match(lines[i])):
                    i += 1
                continue
                
            keep.append(raw)
            i += 1
            
        return "\n".join(keep)
    
    def clean_contextual_chunk(self, contextualized_content: str) -> str:
        """
        Limpa chunk contextualizado para embeddings.
        Preserva contexto no formato [Contexto: ...] se existir.
        
        Args:
            contextualized_content (str): Chunk com possível contexto
            
        Returns:
            str: Chunk limpo otimizado para embeddings
        """
        # Extrai contexto se existir (formato do ContextualEnricher)
        context_match = re.match(r'^(Contexto:.*?)\n---\n(.*)', 
                                contextualized_content, re.DOTALL)
        
        if context_match:
            context_part = context_match.group(1).strip()
            main_content = context_match.group(2).strip()
            
            # Remove headings de aprendizado do conteúdo principal
            main_content = self._remove_learning_headings(main_content)
            
            # Decide como limpar o conteúdo principal
            if self.looks_like_markdown_or_html(main_content):
                clean_main = self.clean_for_embeddings(main_content)
            else:
                clean_main = self.minimal_normalize(main_content)
            
            # Reconstrói com contexto preservado
            return f"[{context_part}] {clean_main}"
        
        # Sem contexto, limpa o conteúdo diretamente
        content = self._remove_learning_headings(contextualized_content)
        
        if self.looks_like_markdown_or_html(content):
            return self.clean_for_embeddings(content)
        else:
            return self.minimal_normalize(content)