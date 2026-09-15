import torch
from models.pinn import PINN
from utils.physics import compute_pde_residual
from utils.data_generator import generate_pde_points, generate_ic_points

# 1. Build the model
model = PINN()

# 2. Test physics loss with random points
x, y, t = generate_pde_points(100)
loss_pde, eta, u, v = compute_pde_residual(model, x, y, t)

# 3. Test initial condition generator
x_ic, y_ic, t_ic, eta_ic, u_ic, v_ic = generate_ic_points(50)

print("All modules work!")
print(f"PINN output shape: {eta.shape}")
print(f"Initial PDE loss: {loss_pde.item():.4f}")
print(f"IC points shape: {eta_ic.shape}")