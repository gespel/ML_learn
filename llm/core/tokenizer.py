import torch

class SimpleTokenizer:
    """Einfacher Character-Level Tokenizer"""
    def __init__(self, text):
        # Alle einzigartigen Zeichen
        self.words = sorted(list(set(text)))
        self.vocab_size = len(self.words)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}
    
    def encode(self, text):
        return [self.word_to_idx.get(word, 0) for word in text]
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            words.append(self.idx_to_word.get(idx, '?'))
        return ' '.join(words)
    
class WordTokenizer:
    def __init__(self, text):
        self.words = re.split("[,;|. \n]", text)
        self.words.append(",")
        self.words.append(".")
        self.words.append("|")
        self.words.append("\n")
        self.words.append(" ")
        self.words.append(";")
        self.words = set(self.words)
        self.words = list(self.words)
        self.vocab_size = len(self.words)

        print(self.vocab_size)
        print(self.words[:10])

        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}
    
    def encode(self, text):
        return [self.word_to_idx.get(word, 0) for word in text]
    
    def decode(self, indices):
        words = []

        for i in indices:
            if isinstance(i, torch.Tensor):
                i = i.item()
            words.append(self.idx_to_word.get(i, '?'))
        return ' '.join(words)

class BPETokenizer:
    """Byte Pair Encoding (BPE) Tokenizer"""
    def __init__(self, text, vocab_size=10000, num_merges=None):
        """
        Args:
            text: Trainingstext
            vocab_size: Maximale Vokabulargröße
            num_merges: Anzahl der Merge-Operationen (wenn None, wird vocab_size verwendet)
        """
        self.vocab_size = vocab_size
        self.num_merges = num_merges or (vocab_size - 256)  # 256 für ASCII-Bytes
        
        # Initialisiere mit Byte-Level Tokens (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}  # Speichere die Merge-Operationen
        
        # Trainiere BPE auf dem Text
        self._train(text)
        
        # Erstelle lookup tables
        self.token_to_idx = {v: i for i, v in self.vocab.items()}
        self.idx_to_token = {i: v for v, i in self.token_to_idx.items()}
    
    def _get_stats(self, ids):
        """Zähle die Häufigkeit aller benachbarten Token-Paare"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge_pair(self, ids, pair, new_token):
        """Ersetze alle Vorkommen eines Paares mit einem neuen Token"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_token)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _train(self, text):
        """Trainiere den BPE Tokenizer"""
        # Konvertiere Text zu Byte-Sequenz
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        
        next_token_id = 256  # Starte nach den 256 ASCII-Tokens
        
        # Führe Merge-Operationen durch
        for i in tqdm.tqdm(range(self.num_merges)):
            stats = self._get_stats(ids)
            
            if not stats:
                break
            
            # Finde das häufigste Paar
            pair = max(stats, key=stats.get)
            
            # Speichere die Merge-Operation
            self.merges[pair] = next_token_id
            
            # Füge neuen Token zum Vokabular hinzu
            self.vocab[next_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # Führe Merge durch
            ids = self._merge_pair(ids, pair, next_token_id)
            next_token_id += 1
            
            if (i + 1) % 100 == 0:
                print(f"BPE Merge {i + 1}/{self.num_merges} | Vokabulgröße: {len(self.vocab)}")
        
        print(f"BPE Training abgeschlossen | Finale Vokabulgröße: {len(self.vocab)}")
    
    def _encode_chunk(self, text_bytes):
        """Kodiere eine Byte-Sequenz mit den gelernten Merges"""
        ids = list(text_bytes)
        
        # Wende alle gelernten Merges in Reihenfolge an
        for pair, token_id in self.merges.items():
            ids = self._merge_pair(ids, pair, token_id)
        
        return ids
    
    def encode(self, text):
        """Kodiere Text zu Token-IDs"""
        text_bytes = text.encode('utf-8')
        return self._encode_chunk(text_bytes)
    
    def decode(self, token_ids):
        """Dekodiere Token-IDs zurück zu Text"""
        text_bytes = b''
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            if token_id in self.vocab:
                text_bytes += self.vocab[token_id]
        
        try:
            return text_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return text_bytes.decode('utf-8', errors='replace')
    
    def _encode_chunk(self, text_bytes):
        """Kodiere eine Byte-Sequenz mit den gelernten Merges"""
        ids = list(text_bytes)
        
        # Wende alle gelernten Merges in Reihenfolge an
        for pair, token_id in self.merges.items():
            ids = self._merge_pair(ids, pair, token_id)
        
        return ids
    
    def encode(self, text):
        """Kodiere Text zu Token-IDs"""
        text_bytes = text.encode('utf-8')
        return self._encode_chunk(text_bytes)
    
    def decode(self, token_ids):
        """Dekodiere Token-IDs zurück zu Text"""
        text_bytes = b''
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            if token_id in self.vocab:
                text_bytes += self.vocab[token_id]
        
        try:
            return text_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return text_bytes.decode('utf-8', errors='replace')