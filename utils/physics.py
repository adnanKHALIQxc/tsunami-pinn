import torch

def compute_pde_residual(model, x, y, t, H=1000.0, g=9.81):
    """
    Computes the residual of the Linear Shallow-Water Equations.
    H = ocean depth (meters)
    g = gravity (m/s^2)
    """
    # 1. Tell PyTorch to track gradients for x, y, and t (Calculus!)
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    # 2. Pass coordinates through the neural network to get predictions
    # The model outputs: wave height (eta), velocity x (u), velocity y (v)
    eta, u, v = model(x, y, t)

    # 3. Calculate the derivatives using Automatic Differentiation
    eta_t = torch.autograd.grad(eta, t, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    eta_x = torch.autograd.grad(eta, x, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    eta_y = torch.autograd.grad(eta, y, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    # 4. The Shallow-Water Equations (The Laws of Physics)
    # We want these equations to equal zero
    eq1 = eta_t + H * (u_x + v_y)       # Conservation of Mass
    eq2 = u_t + g * eta_x               # Conservation of Momentum (X)
    eq3 = v_t + g * eta_y               # Conservation of Momentum (Y)

    # 5. The loss is how much the network violates these equations
    loss_pde = torch.mean(eq1**2) + torch.mean(eq2**2) + torch.mean(eq3**2)
    
    return loss_pde, eta, u, v