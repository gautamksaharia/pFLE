# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 12:28:55 2025

@author: gauta
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import torch.nn as nn
from scipy.fft import fft, fftshift, ifft, fftfreq

torch.manual_seed(0)
np.random.seed(0)

# Domain size
L = 10.0
t0 = 0
tmax = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collocation points (inside the domain)
def generate_collocation_points(N):
    x = torch.rand(N, 1).to(device) * 2 * L - L  # Uniformly from [-L, L]
    t = torch.rand(N, 1).to(device)*tmax - t0              # Uniformly from [t0, tmax]
    return x, t

# Initial condition points (t = t0)
def generate_initial_points(N):
    x = torch.linspace(-L, L, N).view(-1, 1).to(device)
    t = torch.zeros_like(x).to(device)
    return x, t

# Boundary condition points (x = ±L)
def generate_boundary_points(N):
    t = torch.linspace(t0, tmax, N).view(-1, 1).to(device)
    x_left = -L * torch.ones_like(t).to(device)
    x_right = L * torch.ones_like(t).to(device)
    return x_left, x_right, t

plt.scatter(*generate_collocation_points(100))
plt.scatter(*generate_initial_points(100))
xl, xr, tt = generate_boundary_points(100)
plt.scatter(xl, tt)
plt.scatter(xr, tt)
plt.show()



class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = torch.nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            #xt = self.activation(layer(xt))
            xt = torch.sin(layer(xt))
        return self.layers[-1](xt)


# 1-soliton exact solution (from FLE Lashkin Paper :
# Perturbation theory for solitons of the Fokas-Lenells equation: Inverse scattering transform approach
# https://doi.org/10.1103/PhysRevE.103.042203


def u_1soliton(x, t):
    #Generate 1-soliton solution, optionally with noise.
    
    Delta = 1.0
    gamma = 0.1
    
    z = 2 * Delta**2 * (x + t / (4 * Delta**2)) * torch.sin(torch.tensor(gamma))
    phi = 2 * Delta**2 * (x - t / (4 * Delta**2)) * torch.cos(torch.tensor(gamma))
    
    U = (np.sin(gamma) * torch.exp(-1j * phi)) / (1j * Delta * torch.cosh(z + 1j * gamma / 2))
    
    
    # real gaussian
    noise = 0.05 * torch.randn_like(U.real)  # Gaussian N(0,σ²) mean zero variance one
    
    return U


# Initial condition
def r0(x):
    u = u_1soliton(x, t=0)
    return u.real

def m0(x):
    u = u_1soliton(x, t=0)
    return u.imag


xx0, _ = generate_initial_points(100)
rr0 = r0(xx0)
mm0 = m0(xx0)
plt.plot(xx0, rr0)
plt.plot(xx0, mm0)
plt.show()

# Boundary conditions (vanishing boundary condition, zero here)
def bcL(x, t):
    u = u_1soliton(x, t)
    return 0 * u.real

def bcR(x, t):
    u = u_1soliton(x, t)
    return 0 * u.real


# Derivative
def compute_derivatives(f, x, t):
    f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    f_xx = torch.autograd.grad(f_x, x, grad_outputs=torch.ones_like(f_x), create_graph=True)[0]
    f_xxx = torch.autograd.grad(f_xx, x, grad_outputs=torch.ones_like(f_xx), create_graph=True)[0]
    f_t = torch.autograd.grad(f, t, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    f_xt = torch.autograd.grad(f_x, t, grad_outputs=torch.ones_like(f_x), create_graph=True)[0]
    return f_x, f_t, f_xt, f_xxx


e =0.1
# PDE Residuals
def compute_residuals(net, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    output = net(x, t)
    r, m = output[:, 0:1], output[:, 1:2]

    r_x, _, r_xt,r_xxx = compute_derivatives(r, x, t)
    m_x, _, m_xt, m_xxx = compute_derivatives(m, x, t)

    # Fokas Lenells equation with pertubation
    
    # third order
    res_r = r_xt - r - (r**2 + m**2) * m_x - 0.1 * r_xxx
    res_m = m_xt - m + (r**2 + m**2) * r_x - 0.1 * m_xxx
    
    return res_r, res_m


# Loss function
def loss_function(net, x_colloc, t_colloc, n_ib, generate_initial_points, generate_boundary_points):
    # PDE residual loss
    res_r, res_m = compute_residuals(net, x_colloc, t_colloc)
    physics_loss = torch.mean(res_r**2) + torch.mean(res_m**2)
    
    # Initial condition loss
    x0, t0 = generate_initial_points(n_ib)
    out0 = net(x0, t0)
    r0_pred, m0_pred = out0[:, 0:1], out0[:, 1:2]
    # L1 loss and L2 loss
    #ic_loss = torch.mean( torch(r0_pred - r0(x0))) + torch.mean(torch.abs(m0_pred - m0(x0)))
    ic_loss = torch.mean( (r0_pred - r0(x0))**2) + torch.mean((m0_pred - m0(x0))**2)

    # Boundary condition loss
    xL, xR, tb = generate_boundary_points(n_ib)
    outL = net(xL, tb)
    outR = net(xR, tb)
    bc_loss = torch.mean((outL - bcL(xL, tb))**2) + torch.mean((outR - bcR(xR, tb))**2)

    return physics_loss, ic_loss, bc_loss

# train with Adam and L Bfgs
#Adam (Adaptive Moment Estimation), First-order optimizer (uses gradients only).
# Adam is excellent for initial training, especially because PDE residuals can be highly irregular in the beginning.
#L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) Quasi-Newton optimizer (approximates second-order information).
# L-BFGS is excellent for fine-tuning once Adam has gotten close to a good solution


def train_with_adam_then_lbfgs(net, loss_function, generate_collocation_points, n_adam_epochs, n_colloc, n_ib):
    # Adam optimizer initialization
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # Train with Adam for n_adam_epochs
    for epoch in range(n_adam_epochs):
        x, t = generate_collocation_points(n_colloc)
        physics_loss, ic_loss, bc_loss = loss_function(x, t, n_ib)
        loss = 0.1*physics_loss +  7*ic_loss + bc_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"[Adam] Epoch {epoch}, Loss: {loss.item():.4e}, physics_loss:{physics_loss.item():.4e}")
    
    # Define LBFGS closure function
    def closure():
        optimizer_lbfgs.zero_grad()
        x, t = generate_collocation_points(n_colloc)
        physics_loss, ic_loss, bc_loss = loss_function(x, t, n_ib)  # Re-calculate loss for L-BFGS
        loss = 0.1*physics_loss +  7*ic_loss + bc_loss
        loss.backward(retain_graph=True)  # Keep the graph for LBFGS optimization
        return loss

    # Initialize LBFGS optimizer
    optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=15000, line_search_fn="strong_wolfe")
    
    # Run LBFGS optimization
    optimizer_lbfgs.step(closure)

# Plotting function
def plot_outputs(net, x_range=(-L, L), t_range=(t0, tmax), resolution=100):
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
        u_pinn = r + 1j*m

    # Plot the desnity plot of u(x, t)
    plt.figure(figsize=(6, 5))
    plt.contourf(X.numpy(), T.numpy(), u_abs, 100, cmap='viridis')
    plt.colorbar(label='|u(x,t)|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Predicted |u(x,t)|')
    plt.tight_layout()
    plt.show()
    
    # Time propagation Plots
    time_indices = [0, 25, 50, 75, 99]
    time_labels = ["t=0", "t=1.25", "t=2.5", "t=3.75", "t=5"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

    # Set style
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
    })

    for ax, t_idx, label in zip(axes, time_indices, time_labels):
        ax.plot(x, u_abs[:, t_idx], color="black")
        ax.set_title(label)
        ax.set_ylim(0, 1)
        #ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel(r"$|u(x,t)|$")
    for ax in axes:
        ax.set_xlabel("x")
        
    
    axes[0].text(
    0.05, 0.99, "(b)",        # (x, y) in axes fraction coordinates
    transform=axes[0].transAxes,
    fontsize=15,
    fontweight="bold",
    va="top",
    ha="left")
    
    
    #plt.savefig("exp1_snapshotgaussian1.pdf", dpi=400, bbox_inches="tight")  # high-res save
    plt.show()
    
    # Assume u_pinn and tmax are already defined
    nt, N = u_pinn.shape
    x = np.linspace(-10, 10, N)
    t_array = np.linspace(0, tmax, nt)

    kx = fftfreq(N, 1/N)
    kx_shift = fftshift(kx)
    time_index = 70

    # === Compute FFT spectra ===
    psifftlist = []
    for i in range(N):
        psi_fft = np.abs(fftshift(fft(u_pinn[:, i])))
        psi_fft /= psi_fft.max()  # normalize
        psifftlist.append(psi_fft)


    psi_fft1 = np.abs(fftshift(fft(u_pinn[:, time_index])))
    psi_fft1 /= psi_fft1.max()

    # === Plot ===
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    
    t_slice = t_array[time_index]


    # (1,1): Wave field amplitude over x and time
    pcm = ax[0, 0].pcolor(x, t_array, np.abs(u_pinn).T, shading='auto', cmap='plasma')
    ax[0, 0].axhline(y=t_slice, color='white', linestyle='--', linewidth=2, label=f't = {t_slice:.3f}')
    #ax[0, 0].set_title('|u(x, t)|')
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('t')
    fig.colorbar(pcm, ax=ax[0, 0])

    # (1,2): FFT snapshot at a specific time index (70)
    ax[0, 1].plot(x, np.abs(u_pinn[:, time_index]), color='k')
    #ax[0, 1].set_title('FFT at time index 70')
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('|u|')

    # (2,1): FFT evolution over time
    pcm2 = ax[1, 0].pcolor(kx_shift, t_array, np.array(psifftlist), shading='auto', cmap='plasma')
    ax[1, 0].axhline(y=t_slice, color='white', linestyle='--', linewidth=2, label=f't = {t_slice:.3f}')
    #ax[1, 0].set_title('FFT evolution over time')
    ax[1, 0].set_xlabel('k')
    ax[1, 0].set_ylabel('t')
    fig.colorbar(pcm2, ax=ax[1, 0])

    # (2,2): Wave profile at time index 70
    ax[1, 1].plot(kx_shift, psi_fft1, color='k')
    #ax[1, 1].set_title('|u(x)| at time index 70')
    ax[1, 1].set_xlabel('k')
    ax[1, 1].set_ylabel('$\hat{u}$')

    # === Add panel labels (a–d) with color distinction ===
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    panel_colors = ['white', 'black', 'white', 'black']  # a,c = white; b,d = black

    for a, label, color in zip(ax.flatten(), panel_labels, panel_colors):
        a.text(0.02, 0.95, label, transform=a.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='left', color=color)

    plt.tight_layout()
    #plt.savefig("exp1_all_gaussian1.pdf", dpi=400, bbox_inches="tight")
    plt.show()
    
    




# Define the neural network
net = PINN([2, 32, 32, 32, 32, 2])  # Input: (x,t), Output: (r,m)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    net = torch.nn.DataParallel(net)


#net = torch.compile(net)

# Training
train_with_adam_then_lbfgs(
    net=net,
    loss_function=lambda x, t, n: loss_function(net, x, t, n, generate_initial_points, generate_boundary_points),
    generate_collocation_points=generate_collocation_points,
    n_adam_epochs=30000,
    n_colloc=5000,
    n_ib=2000)

# Plotting
plot_outputs(net)

net.eval()  # Set model to eval mode
x_range=(-L, L)
t_range=(t0, tmax)
resolution=100
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
  u_pinn = r + 1j*m

plt.plot( u_abs[:, 0])
plt.plot(u_abs[:, -1])
plt.show()


U0 = u_1soliton(x, t=0)
U5 = u_1soliton(x, t=10)
plt.plot(np.abs(U0))
plt.plot(np.abs(U5))
plt.show()

plt.plot( u_abs[:, 0])
plt.plot(np.abs(U0))
plt.title("compare at t=0")
plt.show()




def spectral_domain1(psi):
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "axes.linewidth": 1.2,
    })

    nt, N = psi.shape
    kx = fftfreq(N, 1/N)
    kx_shift = fftshift(kx)

    
    
    psi_fft_list=[]
    for i in range(N):
        psi_fft = 1j*kx*fftshift(fft(psi[:, i]))
        psi_fft_list.append(psi_fft)
    
    psi_k_abs = np.abs(psi_fft_list)


    # === Plot ===
    fig, ax = plt.subplots(figsize=(6, 6))
    t_array = np.linspace(0, tmax, nt)

    # Smooth pcolormesh for continuous color distribution
    contour = ax.pcolormesh(
        kx_shift, t_array, psi_k_abs,
        shading='auto', cmap='plasma'
    )

    # === Colorbar ===
    cbar = plt.colorbar(contour, ax=ax)
    cbar.ax.tick_params(labelsize=11)

    # === Axis labels and aesthetics ===
    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Time $t$")
    #ax.set_title(r"Spectral evolution $|\tilde{u}(k,t)|$", pad=12)
    ax.tick_params(width=1.2)
    ax.set_xlim(kx_shift.min(), kx_shift.max())
    ax.set_ylim(0, tmax)

    
    # === Save and show ===
    plt.tight_layout()
    #plt.savefig("exp1spectral_evolution_nonoise.pdf", dpi=400, bbox_inches="tight")
    plt.show()


spectra = spectral_domain1(u_pinn)

max_ampl = []
for i in range(u_abs.shape[0]):
    amp = np.max(u_abs[:, i])
    max_ampl.append(amp)
np.array(max_ampl)
plt.plot(np.linspace(0,tmax, u_abs.shape[0]), np.array(max_ampl), label="exper")
plt.plot( np.linspace(0,tmax, u_abs.shape[0]), np.exp(-(np.linspace(0,tmax, u_abs.shape[0]))), "-.",label="fit")
plt.legend()
plt.show()


energy_spatial = []
for i in range(u_abs.shape[0]):
    
    xx = np.linspace(*x_range, resolution)
    dx = xx[1]-xx[0]

    
    dudx = np.gradient(u_pinn[:,i], xx)
    e1 = np.sum(np.abs(dudx)**2)*dx
    
    energy_spatial.append(e1)
plt.plot(np.linspace(0,tmax, u_abs.shape[0]), np.array(energy_spatial), label="exper")
plt.plot( np.linspace(0,tmax, u_abs.shape[0]), np.exp(-(np.linspace(0,tmax, u_abs.shape[0]))), "-.",label="fit")
plt.legend()
plt.show()


momentum_spatial=[]
for i in range(u_abs.shape[0]):
    xx = np.linspace(*x_range, resolution)
    dx = xx[1]-xx[0]
    u = u_pinn[:,i]
    dudx = np.gradient(u_pinn[:,i], xx)
    momentum0 =  np.conjugate(u)*(dudx) - u*np.conjugate(dudx)
    momentum = 0.5*1j*np.sum(momentum0)*dx
    
    momentum_spatial.append(momentum)
plt.plot(np.linspace(0,tmax, u_abs.shape[0]), np.array(momentum_spatial), label="exper")
plt.plot( np.linspace(0,tmax, u_abs.shape[0]), np.exp(-(np.linspace(0,tmax, u_abs.shape[0]))), "-.",label="fit")
plt.legend()
plt.show()


np.save("exp1_gama0.1_max_amplitude_e0.1nonoise", max_ampl)
np.save("exp1_gama0.1_enegy_e0.1nonoise", energy_spatial)
np.save("exp1_gama0.1_momentum_e0.1nonoise", momentum_spatial)
np.save("exp1_gama0.1wavenonoise", u_pinn)


u0_fft = fftshift(fft(u_pinn[:, 0]))
u10_fft = fftshift(fft(u_pinn[:, -1]))
plt.plot(np.abs(u0_fft))
plt.plot(np.abs(u10_fft))
plt.show()



