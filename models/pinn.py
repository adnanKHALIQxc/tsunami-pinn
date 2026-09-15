import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=6, neurons=64):
        super().__init__()
        
        # Input: (x, y, t) = 3 values
        # Output: (eta, u, v) = 3 values
        self.input_layer = nn.Linear(3, neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for _ in range(layers - 2)]
        )
        self.output_layer = nn.Linear(neurons, 3)
        self.activation = nn.Tanh()

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        out = self.output_layer(out)
        
        eta = out[:, 0:1]  # Wave height
        u = out[:, 1:2]    # Velocity X
        v = out[:, 2:3]    # Velocity Y
        
        return eta, u, v