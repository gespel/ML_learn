from pytorch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

n = Net()
print(n)