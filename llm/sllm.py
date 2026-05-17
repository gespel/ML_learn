import argparse
import os

from core.tokenizer import SimpleTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tqdm
import re
import math
import matplotlib.pyplot as plt
from core.modules import SmallLLM



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
):

    plt.ion()
    graph = None

    #TODO: add model parameter store in pt file!

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

    # Tokenizer speichern
    import pickle
    tokenizer_name = f"{tokenizer.vocab_size}v.pkl"
    with open(tokenizer_name, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer gespeichert als: {tokenizer_name}")
    
    # Dataset und DataLoader
    dataset = TextDataset(text, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Datensätze: {len(dataset)}")
    print(f"Batches pro Epoche: {len(dataloader)}")

    vocab_size = tokenizer.vocab_size
    # Modell-Defaults sind bewusst kleiner gewählt, damit Training pro Epoche schneller ist.
    
    # Modell
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
    
    # Checkpoint laden wenn vorhanden
    checkpoint_path = f"{tokenizer.vocab_size}v_{d_model}md_{num_heads}h_{num_layers}l_{d_ff}ff_{max_seq_len}m.pt"
    start_epoch = 1

    old_batch_idx = 0
    last_total_loss = 0

    if os.path.exists(checkpoint_path):
        print(f"Lade Checkpoint von {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(
                "Checkpoint passt nicht zur aktuellen Modellkonfiguration "
                f"(z.B. d_model/Layer/Heads geändert). Starte neu. Details: {e}"
            )
            checkpoint = None

        if checkpoint is not None:
            old_batch_idx = checkpoint.get('batch_idx', old_batch_idx)
            last_total_loss = checkpoint.get('total_loss', 0)
            if checkpoint.get('batch_idx', 0) == 0:
                start_epoch = checkpoint.get('epoch', 0) + 1
                old_batch_idx = 0
                last_total_loss = 0
            else:
                start_epoch = checkpoint.get('epoch', 1)
            print(f"Training wird ab Epoche {start_epoch} fortgesetzt. Letzter Batch-Index: {old_batch_idx}")
    
    # Training loop
    x = []
    y = []
    for epoch in range(start_epoch, num_epochs + 1):
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
            scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / batch_idx
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.8f}")

            #if batch_idx % 100 == 0:
            #    y.append(avg_loss)
            #    x.append(batch_idx)
            #    if graph is not None:
            #        graph.remove()
            #    graph = plt.plot(x, y, color='g')[0]
            #    plt.xlim(x[0], x[-1])
            #    ax = plt.gca()  # get the current axes
            #    ax.relim()      # make sure all the data fits
            #    ax.autoscale()
            #    plt.pause(0.0001)

            #if batch_idx % 1000 == 0:
            #    # Checkpoint speichern
            #    checkpoint = {
            #        'd_model': d_model,
            #        'num_heads': num_heads,
            #        'num_layers': num_layers,
            #        'd_ff': d_ff,
            #        'max_seq_len': max_seq_len,
            #        'vocab_size': vocab_size,
            #        'epoch': epoch,
            #        'model_state_dict': model.state_dict(),
            #        'optimizer_state_dict': optimizer.state_dict(),
            #        'batch_idx': batch_idx,
            #        'total_loss': total_loss
            #    }
            #    torch.save(checkpoint, checkpoint_path)
        
        epoch_loss = total_loss / max(1, len(dataloader))

        # Checkpoint speichern
        checkpoint = {
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'd_ff': d_ff,
            'max_seq_len': max_seq_len,
            'vocab_size': vocab_size,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'batch_idx': 0,
            'total_loss': 0
        }
        torch.save(checkpoint, checkpoint_path)
        last_total_loss = 0
        old_batch_idx = 0
        print(f"Checkpoint gespeichert bei Epoch {epoch}")

        print(f"Epoch {epoch}/{num_epochs} abgeschlossen | Avg Loss: {epoch_loss:.4f}")
        
    # Final Modell speichern (zusätzlich zum Checkpoint)
    torch.save(model.state_dict(), 'small_llm_model.pt')
    print("\nModell gespeichert als: small_llm_model.pt")
    print("Checkpoint gespeichert als: small_llm_checkpoint.pt")

def generate_text(seed: str):
    import pickle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import os
    model_paths = []
    tokenizer_paths = []
    for file in os.listdir("."):
        if file.endswith(".pt"):
            model_paths.append(file)
        elif file.endswith(".pkl"):
            tokenizer_paths.append(file)

    if len(model_paths) > 1:
        print("Multiple models found in this folder. Choose one:")
        for m in enumerate(model_paths):
            print(f"[{m[0]}]: {m[1]}")
        model_path = model_paths[int(input("> "))]
    else:
        model_path = model_paths[0]

    if len(tokenizer_paths) > 1:
        print("Multiple tokenizer found in this folder. Choose one:")
        for m in enumerate(tokenizer_paths):
            print(f"[{m[0]}]: {m[1]}")    
        tokenizer_path = tokenizer_paths[int(input("> "))]
    else:
        tokenizer_path = tokenizer_paths[0]
         


    print(f"Using device: {device}")
    
    # Tokenizer laden
    if os.path.exists(tokenizer_path):
        print(f"Lade Tokenizer von {tokenizer_path}...")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        raise FileNotFoundError(f"Tokenizer nicht gefunden: {tokenizer_path}")

    if os.path.exists(model_path):
        print(f"Lade Checkpoint von {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model = SmallLLM(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            d_ff=checkpoint['d_ff'],
            max_seq_len=checkpoint['max_seq_len']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        seed_tokens = torch.tensor(tokenizer.encode(seed), dtype=torch.long).unsqueeze(0).to(device)
        generated = seed
        temperature = 0.9
        top_k = 10
        
        print(generated, end='', flush=True)

        for _ in range(1000):
            # Begrenze die Sequenzlänge auf max_seq_len
            if seed_tokens.shape[1] > model.max_seq_len:
                seed_tokens = seed_tokens[:, -model.max_seq_len:]
            
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
                print(tokenizer.decode([next_token.item()]), end='', flush=True)
                #generated += tokenizer.decode([next_token.item()])
                seed_tokens = torch.cat([seed_tokens, next_token.unsqueeze(0)], dim=1)
        print()
        #print(generated)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Sten\'s LM',
                    description='This is a small research LLM implementation.',
                    epilog='If you have any questions, please contact me at heimbrodt@uni-potsdam.de')
    
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-g', '--generate', type=str, help='Generate text starting with the given input')
    parser.add_argument('-b', '--batch_size', type=int, help='Set batch size for training.', default=32)
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length for training samples (faster when smaller).')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer model width (smaller = faster).')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer blocks.')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward hidden size.')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum context length the model supports.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')

    args = parser.parse_args()

    if args.train:
       train_model(
           batch_size=args.batch_size,
           seq_len=args.seq_len,
           d_model=args.d_model,
           num_heads=args.num_heads,
           num_layers=args.num_layers,
           d_ff=args.d_ff,
           max_seq_len=args.max_seq_len,
           num_epochs=args.epochs,
       )
    elif args.generate:
        generate_text(args.generate)
    else:
        parser.print_help()

