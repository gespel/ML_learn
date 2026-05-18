from core.modules import *
from core.tokenizer import *
import os
import time
import tqdm
import torch
import socket
import json
import multiprocessing
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import requests


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

def lact_request(payload: dict) -> dict:
    with socket.create_connection(("127.0.0.1", 12853)) as sock:
        sock.sendall((json.dumps(payload) + "\n").encode())

        response = sock.recv(4096)

    return json.loads(response.decode())

def set_gpu_max_frequency(freq: int) -> bool:
    gpu_id = "1002:73AF-1EAE:6905-0000:09:00.0"

    response = lact_request({
        "command": "set_clocks_value",
        "args": {
            "id": gpu_id,
            "command": {
                "type": "max_core_clock",
                "value": freq
            }
        }
    })
    
    if response["status"] != "ok":
        print("set:", response)

    confirm_response = lact_request({
        "command": "confirm_pending_config",
        "args": {
            "command": "confirm"
        }
    })

    if confirm_response["status"] != "ok":
        print("confirm:", confirm_response)

    return True

def get_power_usage() -> float:
    response = lact_request({
        "command": "device_stats",
        "args": {
            "id": "1002:73AF-1EAE:6905-0000:09:00.0"
        }
    })

    if response["status"] != "ok":
        print(response)
    return response["data"]["power"]["average"]

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
    power_measurement_process: multiprocessing.Process = None
):
    text = ""

    for file in os.listdir("../.storage"):
        if file.endswith(".txt"):
            with open(os.path.join("../.storage", file), "r", encoding="utf-8") as f:
                text += f.read() + "\n"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = SimpleTokenizer(text)
    #print(f"Vokabulgröße: {tokenizer.vocab_size}")
    #print(f"Zeichen: {tokenizer.words[:20]} ...")
    
    dataset = TextDataset(text, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #print(f"Datensätze: {len(dataset)}")
    #print(f"Batches pro Epoche: {len(dataloader)}")

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
    if power_measurement_process is not None:
        power_measurement_process.start()
    start_time = time.time()
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

            if batch_idx >= num_of_training_steps:
                end_time = time.time()
                return end_time - start_time
            
def measure_power_loop(queue, interval=0.5):
    while True:
        try:
            power = get_power_usage()
            if power is not None:
                #print(f"Current GPU Power Usage: {power:.2f} W")
                queue.put(power)
        except Exception as e:
            pass
        time.sleep(interval)

def benchmark():    
    training_steps = 2000
    clockspeed = []
    all_runtimes = []
    all_joules = []
    
    overall_power_values = {}


    for i in range(1500, 2700, 100):
        print(f"Now measuring {i} Mhz") 
        set_gpu_max_frequency(i)
        
        q = multiprocessing.Queue()
        power_process = multiprocessing.Process(target=measure_power_loop, args=(q, 0.5))
        
        runtime = train_model(num_of_training_steps=training_steps, power_measurement_process=power_process)
        
        power_process.terminate()
        power_process.join()

        power_values = []
        while not q.empty():
            power_values.append(q.get())
            
        overall_power_values[i] = power_values
        
        it_per_second = training_steps / runtime
        avg_power = sum(power_values) / len(power_values) if power_values else 0
        joules = avg_power * runtime
        clockspeed.append(i)
        all_runtimes.append(runtime)
        all_joules.append(joules)
        print(f"{i} Mhz has a runtime of {runtime:.2f} seconds -> {it_per_second:.2f} step/s | Avg Power: {avg_power:.2f} W | Energy: {joules:.2f} J")

    plt.plot(clockspeed, all_runtimes)
    plt.xlabel("GPU Frequency (MHz)")
    plt.ylabel("Runtime (s)")
    plt.title("GPU Frequency vs Runtime")
    plt.savefig("frequency_vs_runtime.png")
    plt.show()

    plt.plot(clockspeed, all_joules)
    plt.xlabel("GPU Frequency (MHz)")
    plt.ylabel("Energy (Joules)")
    plt.title("GPU Frequency vs Energy Consumption")
    plt.savefig("frequency_vs_energy.png")
    plt.show()

if __name__ == "__main__":
    benchmark()
