# src/main.py

import torch
from model import PINN
from data import generate_collocation_points, generate_initial_points, generate_boundary_points
from physics import loss_function
from train import train_with_adam_then_lbfgs
from plot import plot_outputs

# Set random seed for reproducibility
torch.manual_seed(0)

# Define the neural network
net = PINN([2, 32, 32, 32, 2])  # Input: (x,t), Output: (r,m)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# --- Training ---
train_with_adam_then_lbfgs(
    net=net,
    loss_function=lambda x, t, n: loss_function(
        net, x, t, n, generate_initial_points, generate_boundary_points),
    generate_collocation_points=generate_collocation_points,
    n_adam_epochs=1000,
    n_colloc=1000,
    n_ib=500
)

# --- Plotting ---
plot_outputs(net)
