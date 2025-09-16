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
tmax = 10
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
            xt = self.activation(layer(xt))
            #xt = torch.sin(layer(xt))
        return self.layers[-1](xt)


# 1-soliton exact solution (from FLE Lashkin Paper :
# Perturbation theory for solitons of the Fokas-Lenells equation: Inverse scattering transform approach
# https://doi.org/10.1103/PhysRevE.103.042203

def u_11soliton(x, t):
    Delta = 1.0
    gamma = 1.0
    z = 2 * Delta**2 * (x + t / (4 * Delta**2)) * np.sin(gamma)
    phi = 2 * Delta**2 * (x - t / (4 * Delta**2)) * np.cos(gamma)
    U = (np.sin(gamma) * torch.exp(-1j * phi)) / (1j * Delta * torch.cosh(z + 1j * gamma / 2)) #+ np.random.rand(x.shape)
    return U

def u_1soliton(x, t):
    #Generate 1-soliton solution, optionally with noise.
    
    Delta = 1.0
    gamma = 1.0
    
    z = 2 * Delta**2 * (x + t / (4 * Delta**2)) * torch.sin(torch.tensor(gamma))
    phi = 2 * Delta**2 * (x - t / (4 * Delta**2)) * torch.cos(torch.tensor(gamma))
    
    U = (np.sin(gamma) * torch.exp(-1j * phi)) / (1j * Delta * torch.cosh(z + 1j * gamma / 2))
    
    # None
    #noise = 0.0
    # uniform
    #noise = 0.01 * (2 * torch.rand_like(U.real) - 1)  # uniform [-ε, ε]
    #uniform distribution on the interval [0,1)

    # gaussian
    #noise = 0.01 * torch.randn_like(U.real)  # Gaussian N(0,σ²) mean zero variance one
    
    return U #+ noise

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


# PDE Residuals
def compute_residuals(net, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    output = net(x, t)
    r, m = output[:, 0:1], output[:, 1:2]

    r_x, _, r_xt,r_xxx = compute_derivatives(r, x, t)
    m_x, _, m_xt, m_xxx = compute_derivatives(m, x, t)

    # Fokas Lenells equation with pertubation
    # linear damping perturbation
    
    
    #res_r = r_xt - r - (r**2 + m**2) * m_x #+ 0.1 * r_x
    #res_m = m_xt - m + (r**2 + m**2) * r_x #+ 0.1 * m_x
    
    # third order
    res_r = r_xt - r - (r**2 + m**2) * m_x #- 0.1 * r_xxx
    res_m = m_xt - m + (r**2 + m**2) * r_x #- 0.1 * m_xxx
    
    # e = e0*np.cos(x)
    #res_r = r_xt - r - (r**2 + m**2) * m_x - 0.1*torch.cos(x)*m_x
    #res_m = m_xt - m + (r**2 + m**2) * r_x + 0.1*torch.cos(x)*r_x

    return res_r, res_m


# Loss function
def loss_function(net, x_colloc, t_colloc, n_ib, generate_initial_points, generate_boundary_points):
    # PDE residual loss
    res_r, res_m = compute_residuals(net, x_colloc, t_colloc)
    physics_loss = torch.mean(res_r**2) + torch.mean(res_m**2)
    #physics_loss = torch.mean(torch.abs(res_r)) + torch.mean(torch.abs(res_m))

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
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(n_adam_epochs):
        x, t = generate_collocation_points(n_colloc)
        physics_loss, ic_loss, bc_loss = loss_function(x, t, n_ib)
        loss = 0.1*physics_loss + ic_loss + bc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"[Adam] Epoch {epoch}, Loss: {loss.item():.4e}, physics_loss:{physics_loss.item():.4e}")

    def closure():
        optimizer_lbfgs.zero_grad()
        x, t = generate_collocation_points(n_colloc)
        physics_loss, ic_loss, bc_loss = loss_function(x, t, n_ib) # Re-calculate loss for L-BFGS
        loss = 0.1*physics_loss + ic_loss + bc_loss
        loss.backward(retain_graph=True) # Add retain_graph=True here
        return loss

    optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=15000, line_search_fn="strong_wolfe")
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
    time_labels = ["t=0", "t=2.5", "t=5", "t=7.5", "t=10"]
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
    plt.savefig("soliton_snapshot_third0.pdf", dpi=300, bbox_inches="tight")  # high-res save
    plt.show()
    
    
    # Density Plot
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(5, 7))
    contour = ax.contourf(X.numpy(), T.numpy(), u_abs, 
                          levels=100, cmap='plasma')
    cbar = plt.colorbar(contour, ax=ax)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r" $|u(x,t)|$")
    ax.grid(False)  # cleaner look
    plt.tight_layout()
    plt.savefig("density_third0.pdf", dpi=300, bbox_inches="tight")  # save high-res
    plt.show()



# Define the neural network
net = PINN([2, 32, 32, 32, 32, 2])  # Input: (x,t), Output: (r,m)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
#net = torch.compile(net)

# Training
train_with_adam_then_lbfgs(
    net=net,
    loss_function=lambda x, t, n: loss_function(net, x, t, n, generate_initial_points, generate_boundary_points),
    generate_collocation_points=generate_collocation_points,
    n_adam_epochs=30000,
    n_colloc=5000,
    n_ib=1000)

# Plotting
plot_outputs(net)



# Save specific layers
#torch.save({
#    "layer0": net.layers[0].state_dict(),
#    "layer1": net.layers[1].state_dict()
#}, "partial_layers_pFLE.pth")


# Save all trained weights
#torch.save(net.state_dict(), "base_model_pFLE.pth")




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


def spectral_domain(psi):
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    
    N = psi.shape[1]
    nt = psi.shape[0]
    kx = fftfreq(N, 1/N)
    kx_shift = fftshift(kx)
    psi_fft_list=[]
    for i in range(N):
        psi_fft = fft(psi[:, i])
        psi_fft_shift = fftshift(psi_fft)
        psi_fft_list.append(psi_fft_shift)
    

    fig, ax = plt.subplots(figsize=(5, 7))
    contour = ax.contourf(kx_shift, np.linspace(0,tmax, nt), np.abs(psi_fft_list), 
                      levels=100, cmap='plasma')
    cbar = plt.colorbar(contour, ax=ax)

    ax.set_xlabel(r"wave number $k$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r" $|u(x,t)|$")
    ax.grid(False)  # cleaner look
    plt.tight_layout()
    plt.show()

spectra = spectral_domain(u_abs)
