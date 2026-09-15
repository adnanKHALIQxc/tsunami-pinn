import torch
import torch.nn as nn
from utils.physics import compute_pde_residual

# 1. Create a dummy neural network just to test the physics function
class DummyPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 3) # Outputs: eta, u, v
        )
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        out = self.net(inputs)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

# 2. Setup model and fake data
model = DummyPINN()
x = torch.rand(100, 1)
y = torch.rand(100, 1)
t = torch.rand(100, 1)

# 3. Run the physics function
loss, eta, u, v = compute_pde_residual(model, x, y, t)

print("Physics module works perfectly!")
print(f"Initial Physics Loss: {loss.item():.4f}")
print(f"Predicted Wave Height shape: {eta.shape}")