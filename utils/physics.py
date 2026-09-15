import torch

def compute_pde_residual(model, x, y, t, H=1.0, g=1.0):
    """
    Shallow-Water Equations with NORMALIZED constants.
    H and g are set to 1.0 because the domain is normalized to [0,1].
    Real-world scaling is applied during post-processing.
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    eta, u, v = model(x, y, t)

    eta_t = torch.autograd.grad(eta, t, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    eta_x = torch.autograd.grad(eta, x, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    eta_y = torch.autograd.grad(eta, y, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    eq1 = eta_t + H * (u_x + v_y)
    eq2 = u_t + g * eta_x
    eq3 = v_t + g * eta_y

    loss_pde = torch.mean(eq1**2) + torch.mean(eq2**2) + torch.mean(eq3**2)
    return loss_pde, eta, u, v