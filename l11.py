from torch import nn
import torch
import random

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)

model = Model()
def train(model):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for i in range(10000):
        x = random.random()*10
        x = torch.tensor([[x]], dtype=torch.float)
        y = x * 2
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        print(f"epoch: {i}, loss: {loss}")

train(model)
print(model(torch.tensor([[5.0]])))

