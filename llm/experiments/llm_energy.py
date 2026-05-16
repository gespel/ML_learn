from core.modules import *
from core.tokenizer import *
import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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


def train_model(
    batch_size: int = 32,
    seq_len: int = 256,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    d_ff: int = 1024,
    max_seq_len: int = 256,
    num_epochs: int = 30,
    num_of_training_steps: int = 10000,
):
    text = ""

    for file in os.listdir("../../.storage"):
        if file.endswith(".txt"):
            with open(os.path.join("../../.storage", file), "r", encoding="utf-8") as f:
                text += f.read() + "\n"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"Vokabulgröße: {tokenizer.vocab_size}")
    print(f"Zeichen: {tokenizer.words[:20]} ...")
    
    dataset = TextDataset(text, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Datensätze: {len(dataset)}")
    print(f"Batches pro Epoche: {len(dataloader)}")

    vocab_size = tokenizer.vocab_size

    model = SmallLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(dataloader) * num_epochs)
    )
    loss_fn = nn.CrossEntropyLoss()
        
    # Training loop
    x = []
    y = []
    for epoch in range(0, num_epochs + 1):
        total_loss = 0
        progress = tqdm.tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
            mininterval=0,  # Aktualisiert bei jedem Aufruf
        )
        for batch_idx, (x_batch, y_batch) in enumerate(progress, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            logits = model(x_batch)
            loss = loss_fn(logits.reshape(-1, tokenizer.vocab_size), y_batch.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / batch_idx
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.8f}")

            if progress >= num_of_training_steps:
                return

train_model()