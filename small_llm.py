import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm

# ============================================
# Kleines Language Model (Transformer-basiert)
# ============================================

class BPETokenizer:
    """Byte-Pair Encoding Tokenizer (Subword Token)"""
    def __init__(self, text, vocab_size=300):
        from collections import defaultdict, Counter
        
        # Starte mit Zeichen
        tokens = list(text.lower())
        vocab = set(tokens)
        
        # Merging-Iterationen
        for _ in range(vocab_size - len(vocab)):
            # Finde häufigste Paare
            pairs = defaultdict(int)
            for i in range(len(tokens) - 1):
                pairs[tuple(tokens[i:i+2])] += 1
            
            if not pairs:
                break
            
            best = max(pairs, key=pairs.get)
            vocab.add(''.join(best))
            tokens = self._merge_pair(tokens, best)
        
        self.vocab = sorted(vocab)
        self.token_to_idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx_to_token = {idx: tok for tok, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.vocab)
    
    def _merge_pair(self, tokens, pair):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tuple(tokens[i:i+2]) == pair:
                merged.append(''.join(pair))
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged
    
    def encode(self, text):
        tokens = list(text.lower())
        for _ in range(10):  # Greedy merging
            for pair in self.vocab:
                if len(pair) > 1:
                    tokens = self._merge_pair(tokens, tuple(pair))
        return [self.token_to_idx.get(t, 0) for t in tokens]
    
    def decode(self, indices):
        return ''.join([self.idx_to_token.get(idx, '') for idx in indices])


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
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
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
    def __init__(self, text, tokenizer, seq_len=50):
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
    #text = """Das ist ein kleines Sprachmodell. Es lernt, Text zu generieren. Das Modell basiert auf Transformern. Transformers sind sehr mächtig. Hallo Sten!"""
    
    text = """Baden-Württemberg ([ˌbaːdn̩ˈvʏrtəmbɛrk] ⓘ; Abkürzung BW; amtlich Land Baden-Württemberg) ist ein Land im Südwesten von Deutschland.[6][7] Gemäß seiner Verfassung hat es die Staatsform einer parlamentarischen Republik und ist ein teilsouveräner Gliedstaat der Bundesrepublik Deutschland. Sowohl nach Einwohnerzahl als auch bezüglich der Fläche steht Baden-Württemberg an dritter Stelle der deutschen Länder. Bevölkerungsreichste Stadt ist die Landeshauptstadt Stuttgart, gefolgt von Mannheim und Karlsruhe. Weitere Großstädte sind Freiburg im Breisgau, Heidelberg, Ulm, Heilbronn, Pforzheim und Reutlingen.

Das Land entstand 1952 durch den Zusammenschluss der nach dem Zweiten Weltkrieg gebildeten Länder Württemberg-Baden, (Süd-)Baden und Württemberg-Hohenzollern. Somit steht es in der Tradition der alten Länder Baden und Württemberg sowie der Hohenzollernschen Lande. Baden-Württemberg ist naturräumlich geprägt von seinen Anteilen an der Oberrheinischen Tiefebene und Mittelgebirgen wie dem Schwarzwald, dem Südwestdeutschen Schichtstufenland mit der Schwäbischen Alb und dem Alpenvorland nördlich des Bodensees.

Baden-Württemberg ist das deutsche Land mit den höchsten Exporten (2023),[8] der zweitniedrigsten Arbeitslosenquote (Juli 2024),[9] dem vierthöchsten Bruttoinlandsprodukt (BIP) pro Kopf (2024)[10] sowie den meisten angemeldeten Patenten pro Kopf (2023)[11] und den absolut und relativ höchsten Forschungs- und Entwicklungsausgaben (2021).[12] Die durchschnittliche Lebenserwartung lag im Zeitraum 2018/20 bei 79,9 Jahren für Männer und bei 84,2 Jahren für Frauen, womit beide unter den deutschen Bundesländern jeweils den ersten Rang belegen."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = BPETokenizer(text, vocab_size=1000)
    print(f"Vokabulgröße: {tokenizer.vocab_size}")
    
    # Dataset und DataLoader
    dataset = TextDataset(text, tokenizer, seq_len=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Modell
    model = SmallLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        max_seq_len=100
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 20
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0
        for x_batch, y_batch in dataloader:
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
        
        #if (epoch + 1) % 10 == 0:
        #    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
    
    # Text generieren
    print("\n=== Text Generierung ===")
    model.eval()
    seed_text = "Das ist ein"
    seed_tokens = torch.tensor([tokenizer.encode(seed_text)], device=device)
    
    generated = seed_text
    for _ in range(30):
        with torch.no_grad():
            logits = model(seed_tokens)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            generated += tokenizer.decode([next_token.item()])
            seed_tokens = torch.cat([seed_tokens, next_token], dim=1)
            if seed_tokens.shape[1] > 100:
                seed_tokens = seed_tokens[:, -100:]
    
    print(generated)


if __name__ == "__main__":
    train_model()
