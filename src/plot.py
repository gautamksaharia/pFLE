# src/plot.py

import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Plotting function ---
def plot_outputs(net, x_range=(-5, 5), t_range=(0, 1), resolution=100):
    net.eval()  # Set model to eval mode

    # Create a grid over space and time
    x = torch.linspace(*x_range, resolution).reshape(-1, 1)
    t = torch.linspace(*t_range, resolution).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    # Move to the same device as the model
    device = next(net.parameters()).device
    x_flat = x_flat.to(device)
    t_flat = t_flat.to(device)

    with torch.no_grad():
        out = net(x_flat, t_flat)
        r = out[:, 0].cpu().numpy().reshape(resolution, resolution)
        m = out[:, 1].cpu().numpy().reshape(resolution, resolution)
        u_abs = np.sqrt(r**2 + m**2)

    # Plot the magnitude of u(x, t)
    plt.figure(figsize=(6, 5))
    plt.contourf(X.numpy(), T.numpy(), u_abs, 100, cmap='viridis')
    plt.colorbar(label='|u(x,t)|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Predicted |u(x,t)|')
    plt.tight_layout()
    plt.show()
