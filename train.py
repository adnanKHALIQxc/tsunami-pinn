import torch
import torch.optim as optim
from models.pinn import PINN
from utils.physics import compute_pde_residual
from utils.data_generator import generate_pde_points

EPOCHS = 4000
N_PDE = 2000
LEARNING_RATE = 0.002

model = PINN(layers=6, neurons=64)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("Training PINN (scaled physics, larger net)...")
print("-" * 60)

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    x_pde, y_pde, t_pde = generate_pde_points(N_PDE)
    loss_pde, _, _, _ = compute_pde_residual(model, x_pde, y_pde, t_pde)
    loss_pde.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} | PDE Loss: {loss_pde.item():.6f}")

print("-" * 60)
torch.save(model.state_dict(), "models/pinn_tsunami.pth")
print("Model saved.")