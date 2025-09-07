# src/data.py

import torch

# Domain size
L = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collocation points (inside the domain)
def generate_collocation_points(N):
    x = torch.rand(N, 1).to(device) * 2 * L - L  # Uniformly from [-L, L]
    t = torch.rand(N, 1).to(device)              # Uniformly from [0, 1]
    return x, t

# Initial condition points (t = 0)
def generate_initial_points(N):
    x = torch.linspace(-L, L, N).view(-1, 1).to(device)
    t = torch.zeros_like(x).to(device)
    return x, t

# Boundary condition points (x = ±L)
def generate_boundary_points(N):
    t = torch.linspace(0, 1, N).view(-1, 1).to(device)
    x_left = -L * torch.ones_like(t).to(device)
    x_right = L * torch.ones_like(t).to(device)
    return x_left, x_right, t
