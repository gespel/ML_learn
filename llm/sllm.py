import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tqdm
import re
import math

# ============================================
# Kleines Language Model (Transformer-basiert)
# ============================================

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


class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model muss durch num_heads teilbar sein"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Split in multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.fc_out(output)
        
        return output


class TransformerBlock(nn.Module):
    """Einzelner Transformer Block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = Attention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class SmallLLM(nn.Module):
    """Kleines Language Model"""
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3, d_ff=512, max_seq_len=100):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Output
        logits = self.fc_out(x)
        return logits


class TextDataset(Dataset):
    """Dataset für Textvorhersage"""
    def __init__(self, text, tokenizer, seq_len=256):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return x, y


def train_model():
    """Training Beispiel"""
    # Text für Training
    text = ""

    for file in os.listdir("../.storage"):
        if file.endswith(".txt"):
            with open(os.path.join("../.storage", file), "r", encoding="utf-8") as f:
                text += f.read() + "\n"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"Vokabulgröße: {tokenizer.vocab_size}")
    print(f"Zeichen: {tokenizer.words[:20]} ...")

    #word_tokenizer = WordTokenizer(text)

    # Tokenizer speichern
    import pickle
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer gespeichert als: tokenizer.pkl")
    
    # Dataset und DataLoader
    dataset = TextDataset(text, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Datensätze: {len(dataset)}")
    print(f"Batches pro Epoche: {len(dataloader)}")
    
    # Modell
    model = SmallLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=6,
        d_ff=512,
        max_seq_len=128
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Checkpoint laden wenn vorhanden
    checkpoint_path = 'small_llm_checkpoint.pt'
    start_epoch = 1

    old_batch_idx = 0
    last_total_loss = 0

    if os.path.exists(checkpoint_path):
        print(f"Lade Checkpoint von {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'batch_idx' == 0:
            start_epoch = checkpoint['epoch'] + 1
            old_batch_idx = 0
            last_total_loss = 0
        else:
            start_epoch = checkpoint['epoch']
        old_batch_idx = checkpoint.get('batch_idx', old_batch_idx)
        last_total_loss = checkpoint.get('total_loss', 0)
        print(f"Training wird ab Epoche {start_epoch} fortgesetzt. Letzter Batch-Index: {old_batch_idx}")
    
    # Training loop
    num_epochs = 100
    for epoch in range(start_epoch, num_epochs):
        total_loss = last_total_loss
        progress = tqdm.tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
            mininterval=0,  # Aktualisiert bei jedem Aufruf
        )
        for batch_idx, (x_batch, y_batch) in enumerate(progress, start=1):
            if batch_idx <= old_batch_idx:
                avg_loss = total_loss / batch_idx
                progress.set_postfix(loss=f"{avg_loss:.4f} (übersprungen)")
                continue  # Überspringe bereits trainierte Batches
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            logits = model(x_batch)
            loss = loss_fn(logits.reshape(-1, tokenizer.vocab_size), y_batch.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / batch_idx
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

            if batch_idx % 1000 == 0:
                # Checkpoint speichern
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'batch_idx': batch_idx,
                    'total_loss': total_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint gespeichert bei Epoch {epoch}, Batch {batch_idx}")
        
        epoch_loss = total_loss / max(1, len(dataloader))

        # Checkpoint speichern
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'batch_idx': 0,
            'total_loss': 0
        }
        torch.save(checkpoint, checkpoint_path)
        last_total_loss = 0
        old_batch_idx = 0
        print(f"Checkpoint gespeichert bei Epoch {epoch + 1}")

        print(f"Epoch {epoch}/{num_epochs} abgeschlossen | Avg Loss: {epoch_loss:.4f}")
    
    # Text generieren
    print("\n=== Text Generierung ===")
    model.eval()
    seed_text = "Baden"
    seed_tokens = torch.tensor(tokenizer.encode(seed_text), dtype=torch.long).unsqueeze(0).to(device)
    
    generated = seed_text
    temperature = 0.9
    top_k = 10
    
    for _ in range(100):
        with torch.no_grad():
            logits = model(seed_tokens)
            next_token_logits = logits[0, -1, :] / temperature

            if top_k is not None and top_k > 0:
                values, indices = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[indices] = values
                next_token_logits = filtered_logits

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated += tokenizer.decode([next_token.item()])
            seed_tokens = torch.cat([seed_tokens, next_token.unsqueeze(0)], dim=1)
            if seed_tokens.shape[1] > 100:
                seed_tokens = seed_tokens[:, -100:]
    print(generated)
    
    # Final Modell speichern (zusätzlich zum Checkpoint)
    torch.save(model.state_dict(), 'small_llm_model.pt')
    print("\nModell gespeichert als: small_llm_model.pt")
    print("Checkpoint gespeichert als: small_llm_checkpoint.pt")
    


def generate_text(input: str):
    import pickle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = 'small_llm_checkpoint.pt'
    tokenizer_path = 'tokenizer.pkl'

    print(f"Using device: {device}")
    
    # Tokenizer laden
    if os.path.exists(tokenizer_path):
        print(f"Lade Tokenizer von {tokenizer_path}...")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        raise FileNotFoundError(f"Tokenizer nicht gefunden: {tokenizer_path}")

    model = SmallLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=6,
        d_ff=512,
        max_seq_len=128
    ).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Lade Checkpoint von {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        seed_tokens = torch.tensor(tokenizer.encode(input), dtype=torch.long).unsqueeze(0).to(device)
        generated = input
        temperature = 0.9
        top_k = 10
        
        for _ in range(1000):
            # Begrenze die Sequenzlänge auf max_seq_len
            if seed_tokens.shape[1] > 99:
                seed_tokens = seed_tokens[:, -99:]
            
            with torch.no_grad():
                logits = model(seed_tokens)
                next_token_logits = logits[0, -1, :] / temperature

                if top_k is not None and top_k > 0:
                    values, indices = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                    filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                    filtered_logits[indices] = values
                    next_token_logits = filtered_logits

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated += tokenizer.decode([next_token.item()])
                seed_tokens = torch.cat([seed_tokens, next_token.unsqueeze(0)], dim=1)
        print(generated)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Sten\'s LM',
                    description='This is a small research LLM implementation.',
                    epilog='If you have any questions, please contact me at heimbrodt@uni-potsdam.de')
    
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-g', '--generate', type=str, help='Generate text starting with the given input')

    args = parser.parse_args()

    if args.train:
       train_model()
    elif args.generate:
        generate_text(args.generate)
    else:
        parser.print_help()