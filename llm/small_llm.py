import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tqdm

# ============================================
# Kleines Language Model (Transformer-basiert)
# ============================================

class SimpleTokenizer:
    """Einfacher Character-Level Tokenizer"""
    def __init__(self, text):
        # Alle einzigartigen Zeichen
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        chars = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            chars.append(self.idx_to_char.get(idx, '?'))
        return ''.join(chars)


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
    
    text = """Baden-Württemberg ist ein Land im Südwesten von Deutschland. Gemäß seiner Verfassung hat es die Staatsform einer parlamentarischen Republik und ist ein teilsouveräner Gliedstaat der Bundesrepublik Deutschland. Sowohl nach Einwohnerzahl als auch bezüglich der Fläche steht Baden-Württemberg an dritter Stelle der deutschen Länder. Bevölkerungsreichste Stadt ist die Landeshauptstadt Stuttgart, gefolgt von Mannheim und Karlsruhe. Weitere Großstädte sind Freiburg im Breisgau, Heidelberg, Ulm, Heilbronn, Pforzheim und Reutlingen. Das Land entstand 1952 durch den Zusammenschluss der nach dem Zweiten Weltkrieg gebildeten Länder Württemberg-Baden, (Süd-)Baden und Württemberg-Hohenzollern. Somit steht es in der Tradition der alten Länder Baden und Württemberg sowie der Hohenzollernschen Lande. Baden-Württemberg ist naturräumlich geprägt von seinen Anteilen an der Oberrheinischen Tiefebene und Mittelgebirgen wie dem Schwarzwald, dem Südwestdeutschen Schichtstufenland mit der Schwäbischen Alb und dem Alpenvorland nördlich des Bodensees. Baden-Württemberg ist das deutsche Land mit den höchsten Exporten der zweitniedrigsten Arbeitslosenquote , dem vierthöchsten Bruttoinlandsprodukt (BIP) pro Kopf (2024)[10] sowie den meisten angemeldeten Patenten pro Kopf (2023)[11] und den absolut und relativ höchsten Forschungs- und Entwicklungsausgaben (2021).Die durchschnittliche Lebenserwartung lag im Zeitraum 2018/20 bei 79,9 Jahren für Männer und bei 84,2 Jahren für Frauen, womit beide unter den deutschen Bundesländern jeweils den ersten Rang belegen."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"Vokabulgröße: {tokenizer.vocab_size}")
    print(f"Zeichen: {tokenizer.chars[:20]} ...")
    
    # Dataset und DataLoader
    dataset = TextDataset(text, tokenizer, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Modell
    model = SmallLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        max_seq_len=100
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 5000
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
    seed_text = "Baden"
    seed_tokens = torch.tensor(tokenizer.encode(seed_text), dtype=torch.long).unsqueeze(0).to(device)
    
    generated = seed_text
    
    for _ in range(100):
        with torch.no_grad():
            logits = model(seed_tokens)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated += tokenizer.decode([next_token.item()])
            seed_tokens = torch.cat([seed_tokens, next_token.unsqueeze(0)], dim=1)
            if seed_tokens.shape[1] > 100:
                seed_tokens = seed_tokens[:, -100:]
    print(generated)
    
    # Modell speichern
    torch.save(model.state_dict(), 'small_llm_model.pt')
    print("\nModell gespeichert als: small_llm_model.pt")


if __name__ == "__main__":
    train_model()
