import torch
import torch.optim as optim
from models.pinn import PINN
from utils.physics import compute_pde_residual
from utils.data_generator import generate_pde_points, generate_ic_points

# 1. Configuration
EPOCHS = 2000
N_PDE = 1000        # Points to check physics
N_IC = 500          # Points to check initial condition
LEARNING_RATE = 0.001

# 2. Initialize model and optimizer
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting PINN Training...")
print("-" * 60)

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # --- Loss 1: Physics (PDE residual) ---
    x_pde, y_pde, t_pde = generate_pde_points(N_PDE)
    loss_pde, _, _, _ = compute_pde_residual(model, x_pde, y_pde, t_pde)

    # --- Loss 2: Initial Condition ---
    x_ic, y_ic, t_ic, eta_ic, u_ic, v_ic = generate_ic_points(N_IC)
    eta_pred, u_pred, v_pred = model(x_ic, y_ic, t_ic)
    
    loss_ic = torch.mean((eta_pred - eta_ic)**2) + \
              torch.mean((u_pred - u_ic)**2) + \
              torch.mean((v_pred - v_ic)**2)

    # --- Total Loss ---
    loss = loss_pde + loss_ic

    # Backpropagation (Calculus!)
    loss.backward()
    optimizer.step()

    # Print progress every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Total: {loss.item():.4f} | PDE: {loss_pde.item():.4f} | IC: {loss_ic.item():.4f}")

print("-" * 60)
print("Training Complete!")

# Save the trained model
torch.save(model.state_dict(), "models/pinn_tsunami.pth")
print("Model saved to models/pinn_tsunami.pth")