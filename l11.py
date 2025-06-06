from torch import nn
import torch
import tqdm
import random

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)

model = Model()
print(f"Version: {torch.__version__}, GPU: {torch.cuda.is_available()}, NUM_GPU: {torch.cuda.device_count()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in tqdm.tqdm(range(100000)):
        a = random.random()
        b = random.random()
        x = torch.tensor([[a, b]], dtype=torch.float)
        y = torch.tensor([[a+b]], dtype=torch.float)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

train(model)
while True:
    inputa = input("A: ")
    inputb = input("B: ")
    print(model(torch.tensor([[float(inputa), float(inputb)]])).item())

