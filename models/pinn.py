import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=6, neurons=64):
        super().__init__()
        
        self.input_layer = nn.Linear(3, neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for _ in range(layers - 2)]
        )
        self.output_layer = nn.Linear(neurons, 3)
        self.activation = nn.Tanh()
        
        # Better initialization for PINNs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def initial_condition(self, x, y):
        """Gaussian bump representing the initial earthquake."""
        A = 1.0
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return A * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        raw = self.output_layer(out)
        
        # Hard IC constraint: at t=0, eta = Gaussian, u = v = 0
        eta_ic = self.initial_condition(x, y)
        
        eta = eta_ic + t * raw[:, 0:1]
        u = t * raw[:, 1:2]
        v = t * raw[:, 2:3]
        
        return eta, u, v