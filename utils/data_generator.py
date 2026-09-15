import torch

def generate_pde_points(n_points):
    """
    Random coordinates inside the domain to test physics.
    x, y in [0, 1] (space) and t in [0, 1] (time)
    """
    x = torch.rand(n_points, 1)
    y = torch.rand(n_points, 1)
    t = torch.rand(n_points, 1)
    return x, y, t

def generate_ic_points(n_points):
    """
    Points at t=0 representing the initial earthquake (Gaussian bump).
    """
    x = torch.rand(n_points, 1)
    y = torch.rand(n_points, 1)
    t = torch.zeros(n_points, 1)
    
    A = 1.0             # Amplitude of the initial wave
    x0, y0 = 0.5, 0.5   # Center of the earthquake
    sigma = 0.1         # Width of the wave
    
    eta_ic = A * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    u_ic = torch.zeros_like(eta_ic)
    v_ic = torch.zeros_like(eta_ic)
    
    return x, y, t, eta_ic, u_ic, v_ic