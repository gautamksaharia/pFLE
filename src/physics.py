# src/physics.py

import torch
import numpy as np
from model import PINN


# Parameters
L = 5.0  # Domain size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1-soliton exact solution (from FLE Lashkin Paper :
# Perturbation theory for solitons of the Fokas-Lenells equation: Inverse scattering transform approach
# https://doi.org/10.1103/PhysRevE.103.042203

def u_1soliton(x, t):
    Delta = 1.0
    gamma = 1.0
    z = 2 * Delta**2 * (x + t / (4 * Delta**2)) * np.sin(gamma)
    phi = 2 * Delta**2 * (x - t / (4 * Delta**2)) * np.cos(gamma)
    U = (np.sin(gamma) * torch.exp(-1j * phi)) / (1j * Delta * torch.cosh(z + 1j * gamma / 2))
    return U


# Initial condition 
def r0(x):
    u = u_1soliton(x, t=0)
    return u.real

def m0(x):
    u = u_1soliton(x, t=0)
    return u.imag


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
    f_t = torch.autograd.grad(f, t, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    f_xt = torch.autograd.grad(f_x, t, grad_outputs=torch.ones_like(f_x), create_graph=True)[0]
    return f_x, f_t, f_xt


# PDE Residuals
def compute_residuals(net, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    output = net(x, t)
    r, m = output[:, 0:1], output[:, 1:2]

    r_x, _, r_xt = compute_derivatives(r, x, t)
    m_x, _, m_xt = compute_derivatives(m, x, t)

    # Fokas Lenells equation with pertubation
    # linear damping perturbation
    res_r = r_xt - r - (r**2 + m**2) * m_x + 0.1 * r_x
    res_m = m_xt - m + (r**2 + m**2) * r_x + 0.1 * m_x

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
    ic_loss = torch.mean((r0_pred - r0(x0))**2) + torch.mean((m0_pred - m0(x0))**2)

    # Boundary condition loss
    xL, xR, tb = generate_boundary_points(n_ib)
    outL = net(xL, tb)
    outR = net(xR, tb)
    bc_loss = torch.mean((outL - bcL(xL, tb))**2) + torch.mean((outR - bcR(xR, tb))**2)

    return physics_loss, ic_loss, bc_loss
