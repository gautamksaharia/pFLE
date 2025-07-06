# src/model.py
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            xt = self.activation(layer(xt))
        return self.layers[-1](xt)
