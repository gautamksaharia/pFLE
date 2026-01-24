import torch
import numpy as np
import torch.nn as nn
import math
from scipy.interpolate import RegularGridInterpolator
import argparse
from scipy.fft import fft, fftshift, ifft, fftfreq

# ============================================================
#               PARSER FOR JOB ARRAY
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--realization", type=int, required=True)
parser.add_argument("--adam-epochs", type=int, default=20000)
parser.add_argument("--n-colloc", type=int, default=5000)
parser.add_argument("--n-ib", type=int, default=2000)
parser.add_argument("--out-dir", type=str, default="outputs_100")
parser.add_argument("--resolution", type=int, default=500)
args = parser.parse_args()



import os
out_dir = args.out_dir if hasattr(args, "out_dir") else "outputs_100"
os.makedirs(out_dir, exist_ok=True)
prefix = os.path.join(out_dir, f"R{args.realization:03d}_")





real_id = args.realization
prefix = f"{args.out_dir}/R{real_id:03d}_"

print(f"Running realization = {real_id}")
print(f"Outputs saved with prefix = {prefix}")

# ============================================================
#                 DEVICE SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
#             DOMAIN & PROBLEM PARAMETERS
# ============================================================
L = 10.0
t0 = 0
tmax = 5

# ============================================================
#             COLL / BC / IC POINT GENERATORS
# ============================================================
def generate_collocation_points(N):
    x = torch.rand(N, 1).to(device) * 2 * L - L
    t = torch.rand(N, 1).to(device) * tmax
    return x, t

def generate_initial_points(N):
    x = torch.linspace(-L, L, N).view(-1, 1).to(device)
    t = torch.zeros_like(x)
    return x, t

def generate_boundary_points(N):
    t = torch.linspace(t0, tmax, N).view(-1, 1).to(device)
    x_left = -L * torch.ones_like(t)
    x_right = L * torch.ones_like(t)
    return x_left, x_right, t

# ============================================================
#                     PINN MODEL
# ============================================================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                     for i in range(len(layers)-1)])

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            xt = self.activation(layer(xt))
        return self.layers[-1](xt)

# ============================================================
#          SOLITON + NOISE  (ONE REALIZATION)
# ============================================================
def u_1soliton(x, t):
    Delta = 1.0
    gamma = 1.0

    z = 2 * Delta**2 * (x + t/(4*Delta**2)) * torch.sin(torch.tensor(gamma))
    phi = 2 * Delta**2 * (x - t/(4*Delta**2)) * torch.cos(torch.tensor(gamma))

    U = (np.sin(gamma) * torch.exp(-1j * phi)) / \
        (1j * Delta * torch.cosh(z + 1j * gamma/2))

    return U

def r0(x):
    return u_1soliton(x, 0).real

def m0(x):
    return u_1soliton(x, 0).imag

# ============================================================
#           AUTOMATIC DIFFERENTIATION UTILITIES
# ============================================================
def compute_derivatives(f, x, t):
    f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                              create_graph=True)[0]
    f_xx = torch.autograd.grad(f_x, x, grad_outputs=torch.ones_like(f_x),
                               create_graph=True)[0]
    f_xxx = torch.autograd.grad(f_xx, x, grad_outputs=torch.ones_like(f_xx),
                                create_graph=True)[0]
    f_t = torch.autograd.grad(f, t, grad_outputs=torch.ones_like(f),
                              create_graph=True)[0]
    f_xt = torch.autograd.grad(f_x, t, grad_outputs=torch.ones_like(f_x),
                               create_graph=True)[0]
    return f_x, f_t, f_xt, f_xxx




# ------------------------------------------------------------
# 1. Parameters (from your model / paper)
# ------------------------------------------------------------
sigma = 0.1        # variance of epsilon_0
q0 = 2*np.pi       # spatial forcing wave number
omega_c = 4.0      # temporal cutoff frequency
B0 = 1.0           # amplitude of temporal spectrum

# ------------------------------------------------------------
# 2. Create grids for noise
# ------------------------------------------------------------
Nx_eps = 300
Nt_eps = 300

x_eps = np.linspace(-L, L, Nx_eps)
t_eps = np.linspace(t0, tmax, Nt_eps)
dt = t_eps[1] - t_eps[0]

# ------------------------------------------------------------
# 3. Spatial part: ε(x) = ε0 cos(q0 x + θ)
# ------------------------------------------------------------
eps0 = np.random.normal(0, sigma)      # ε0 ~ N(0, σ²)
theta = np.random.uniform(0, 2*np.pi)  # random phase

eps_x = eps0 * np.cos(q0 * x_eps + theta)     # shape: (Nx_eps,)

# ------------------------------------------------------------
# 4. Temporal part: Gaussian spectrum B̂(ω) = B0 * exp(-(ω/ωc)^2)
# ------------------------------------------------------------
omega = fftfreq(Nt_eps, d=dt) * 2*np.pi        # angular frequencies
B_hat_sqrt = np.sqrt(B0 * np.exp(-(omega/omega_c)**2))

# White noise in frequency domain
xi = (np.random.normal(size=Nt_eps) +
      1j*np.random.normal(size=Nt_eps))

eta_t = np.real(ifft(B_hat_sqrt * xi))         # temporal noise, shape: (Nt_eps,)

# ------------------------------------------------------------
# 5. Construct full ε(x,t)
# ------------------------------------------------------------
eps_xt = np.outer(eta_t, eps_x)   # shape: (Nt_eps, Nx_eps)

# ------------------------------------------------------------
# 6. Interpolator for PINN
# ------------------------------------------------------------
eps_interp = RegularGridInterpolator((t_eps, x_eps), eps_xt)





# ============================================================
#           PDE RESIDUAL (with multiplicative noise)
# ============================================================
def compute_residuals(net, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    out = net(x, t)
    r, m = out[:, 0:1], out[:, 1:2]

    r_x, r_t, r_xt, r_xxx = compute_derivatives(r, x, t)
    m_x, m_t, m_xt, m_xxx = compute_derivatives(m, x, t)

    # IMPORTANT: use ε(x,t) interpolated to training points
    # ----------------------------------------------------
    xt_np = torch.cat([t.detach().cpu(), x.detach().cpu()], dim=1).numpy()
    eps_vals = eps_interp(xt_np)
    eps_vals = torch.tensor(eps_vals, dtype=torch.float32).view(-1,1).to(device)

    alpha =0.1
    f = 1- torch.exp(-alpha*t)


    res_r = r_xt - r - (r**2 + m**2) * m_x - f*eps_vals * m_x
    res_m = m_xt - m + (r**2 + m**2) * r_x + f*eps_vals * r_x
    return res_r, res_m

# ============================================================
#                    LOSS FUNCTION
# ============================================================
def loss_function(net, x_colloc, t_colloc, n_ib):
    res_r, res_m = compute_residuals(net, x_colloc, t_colloc)
    physics_loss = torch.mean(res_r**2) + torch.mean(res_m**2)

    x0, t0 = generate_initial_points(n_ib)
    out0 = net(x0, t0)
    r0_pred, m0_pred = out0[:, 0:1], out0[:, 1:2]
    ic_loss = torch.mean((r0_pred - r0(x0))**2) + torch.mean((m0_pred - m0(x0))**2)

    xL, xR, tb = generate_boundary_points(n_ib)
    outL = net(xL, tb)
    outR = net(xR, tb)
    bc_loss = torch.mean(outL**2) + torch.mean(outR**2)

    return physics_loss, ic_loss, bc_loss

# ============================================================
#          TRAINING: ADAM + L-BFGS
# ============================================================
def train(net, n_epochs, n_colloc, n_ib):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    for e in range(n_epochs):
        x,t = generate_collocation_points(n_colloc)
        p, i, b = loss_function(net, x, t, n_ib)
        loss = p + i + b

        opt.zero_grad()
        loss.backward()
        opt.step()

        if e % 20 == 0:
            print(f"[Real {real_id}] Adam {e}: loss={loss.item():.3e}")

    # LBFGS
    opt2 = torch.optim.LBFGS(net.parameters(), max_iter=10000,
                             line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad()
        x,t = generate_collocation_points(n_colloc)
        p, i, b = loss_function(net, x, t, n_ib)
        loss = p + i + b
        loss.backward()
        return loss

    print(f"[Real {real_id}] Starting LBFGS")
    opt2.step(closure)

# ============================================================
#                       MAIN EXECUTION
# ============================================================
net = PINN([2,32,32,32,32,2]).to(device)

train(net, args.adam_epochs, args.n_colloc, args.n_ib)

# ============================================================
#            SAVE RESULTS FOR THIS REALIZATION
# ============================================================
net.eval()
resolution = args.resolution

x = torch.linspace(-L, L, resolution).reshape(-1,1)
t = torch.linspace(t0, tmax, resolution).reshape(-1,1)
X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

x_flat = X.reshape(-1,1).to(device)
t_flat = T.reshape(-1,1).to(device)

with torch.no_grad():
    out = net(x_flat, t_flat)
    r = out[:,0].cpu().numpy().reshape(resolution,resolution)
    m = out[:,1].cpu().numpy().reshape(resolution,resolution)

u_pinn = r + 1j*m
u_abs = np.abs(u_pinn)

np.save(prefix + "u_complex.npy", u_pinn)
np.save(prefix + "u_abs.npy", u_abs)
np.save(prefix + "r.npy", r)
np.save(prefix + "m.npy", m)

print(f"[Real {real_id}] Saved → {prefix}")

