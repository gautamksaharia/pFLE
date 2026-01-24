# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 03:05:45 2025

@author: gauta
"""


import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# -----------------------------
# Parameters
# -----------------------------
x_range = (-10, 10)
resolution = 500
tmax = 5
#selected_indices = [3,7,11, 15, 19, 23, 27]
selected_indices = [-1,-5, -9, -13, -17, -21, -25]


# Folder containing all realizations
folder = "outputs_100_10dec"
# Load files
files = sorted(glob(os.path.join(folder, "r*.npy")))
print(f"Found {len(files)} files.")




# Spatial grid (constant for all)
x = np.linspace(*x_range, resolution)
dx = x[1] - x[0]

# Wave numbers (FFT frequencies)
k = np.fft.fftshift(np.fft.fftfreq(resolution, d=dx))

# -----------------------------
# Helper functions
# -----------------------------
def compute_energy(u):
    """Compute spatial derivative energy for each timestep."""
    dudx = np.gradient(u, x, axis=0)  # vectorized gradient
    energy = np.sum(np.abs(dudx)**2, axis=0) * dx
    return energy

def compute_max_amplitude(u):
    """Compute max(|u|) per timestep."""
    return np.max(np.abs(u), axis=0)


def compute_energy_fourier(u):
    """
    Fourier-domain energy:
    E_k(t) = sum |u_hat(k,t)|^2 dk
    Also returns the full spectrum E(k,t).
    """
    u_hat = np.fft.fftshift(np.fft.fft(u, axis=0), axes=0)
    k = np.fft.fftshift (np.fft.fftfreq(500, 0.01) * 2*np.pi)
    spectrum = np.log10(np.abs(u_hat)**2 +1e-12)                   # shape: (Nx, Nt)
    total_Ek = np.sum(spectrum, axis=0) * (k[1]-k[0])
    return total_Ek, spectrum


def plot_time_series(t, y, title, ylabel):
    plt.figure(figsize=(6,4))
    plt.plot(t, y, label=title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Process selected realizations
# -----------------------------
Energy = []
Amplitudes = []
Times = []

Energy_fourier = []
Spectrum_list = []
t_list = []


for idx in selected_indices:
    print(f"Processing file: {files[idx]}")
    
    u = np.load(files[idx])
    Nt = u.shape[1]
    t = np.linspace(0, tmax, Nt)

    energy = compute_energy(u)
    ampl   = compute_max_amplitude(u)
    
    Ek_tot, spectrum = compute_energy_fourier(u)

    Energy_fourier.append(Ek_tot)
    Spectrum_list.append(spectrum)
    t_list.append(t)
    


    Energy.append(energy)
    Amplitudes.append(ampl)
    Times.append(t)

print("Processing complete.")



Energy_fourier = np.array(Energy_fourier)





# -----------------------------
# Compute mean and std
# -----------------------------
energy_mean = np.mean(Energy, axis=0)
energy_std  = np.std(Energy, axis=0)

ampl_mean   = np.mean(Amplitudes, axis=0)
ampl_std    = np.std(Amplitudes, axis=0)




fig = plt.figure(figsize=(18, 9))

# Outer grid: 2 row, 1 columns → two 6x6 areas
outer = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
# ---- Left: 3x3 (6x6) ----
gs_left = outer[0].subgridspec(2, 1, wspace=0.3, hspace=0.3)
# ---- Right: 2x2 (6x6) ----
gs_right = outer[1].subgridspec(2, 2, wspace=0.3, hspace=0.3)


axes0 = fig.add_subplot(gs_left[0, 0])
axes1 = fig.add_subplot(gs_left[1, 0])



# Create a figure with 2 subplots
#fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

# Plot energy on the first subplot
#for i, idx in enumerate(selected_indices):
 #   axes[0].plot(t, Energy[i], alpha=0.6, lw=1.0, color='blue')

# Plot mean energy line
axes0.plot(t, energy_mean, color='black', lw=2.5, label="Mean energy")

# Add shaded region for ±1 std
axes0.fill_between(t, energy_mean - energy_std, energy_mean + energy_std,
                     color="blue", alpha=0.3, label="± std")

# Customize the first plot (Energy)
axes0.set_ylabel("Energy", fontsize=16 ,fontweight="bold")
axes0.set_ylim(3.7, 4.3)
axes0.legend(loc='upper right', fontsize=14)
axes0.spines['bottom'].set_linewidth(2.5)
axes0.spines['left'].set_linewidth(2.5)
axes0.spines['top'].set_linewidth(2.5)
axes0.spines['right'].set_linewidth(2.5)
axes0.tick_params(axis='both', direction='in', length=6, width=2, labelsize=18)

# Plot amplitudes on the second subplot
#for i, idx in enumerate(selected_indices):
 #   axes[1].plot(t, Amplitudes[i], alpha=0.6, lw=1.0, color='Red')

# Plot mean amplitude line
axes1.plot(t, ampl_mean, color='black', lw=2.5, label="Mean amplitude")

# Add shaded region for ±1 std
axes1.fill_between(t, ampl_mean - ampl_std, ampl_mean + ampl_std,
                     color="Red", alpha=0.3, label="± std")

# Customize the second plot (Amplitude)
axes1.set_xlabel("Time t", fontsize=16,fontweight="bold")
axes1.set_ylabel("Amplitude", fontsize=16, fontweight="bold")
axes1.set_ylim(0.8, 1.1)
axes1.legend(loc='upper right', fontsize=14 )
axes1.spines['bottom'].set_linewidth(2.5)
axes1.spines['left'].set_linewidth(2.5)
axes1.spines['top'].set_linewidth(2.5)
axes1.spines['right'].set_linewidth(2.5)
axes1.tick_params(axis='both', direction='in', length=6, width=2, labelsize=18)











u0 = np.load(files[-1])
u1 = np.load(files[-5])
u2 = np.load(files[-9])
u3 = np.load(files[-13])


u_add = np.array([u0,u1,u2, u3])
u = np.mean(u_add, axis=0)





idx_list = [20, 150, 250, 410]
labels = ['(a)', '(b)', '(c)', '(d)']

k = np.fft.fftshift(np.fft.fftfreq(500, d=1) * 2 * np.pi)


ax0 = fig.add_subplot(gs_right[0, 0])
ax1 = fig.add_subplot(gs_right[0, 1])
ax2 = fig.add_subplot(gs_right[1, 0])
ax3 = fig.add_subplot(gs_right[1, 1])




u_slice = u[:, idx_list[0]]
spectrum = np.fft.fftshift(np.abs(np.fft.fft(u_slice)))
# Avoid log(0)
spec_log = np.log10(spectrum + 1e-12)
ax0.plot(k, spec_log, lw=2, color='black')
ax0.set_title(f"{labels[0]}  t = {idx_list[0]/100}", loc='left' ,fontweight="bold", fontsize=14)
ax0.set_xlabel("Wavenumber k", fontsize=14 ,fontweight="bold")
    #ax.set_ylabel(r"$\log_{10} E(k)$")
ax0.spines['bottom'].set_linewidth(2.5)
ax0.spines['left'].set_linewidth(2.5)
ax0.spines['top'].set_linewidth(2.5)
ax0.spines['right'].set_linewidth(2.5)
ax0.tick_params(direction='in', length=6, width=2, labelsize=18)











u_slice = u[:, idx_list[1]]
spectrum = np.fft.fftshift(np.abs(np.fft.fft(u_slice)))
# Avoid log(0)
spec_log = np.log10(spectrum + 1e-12)
ax1.plot(k, spec_log, lw=2, color='black')
ax1.set_title(f"{labels[1]}  t = {idx_list[1]/100}", loc='left',fontweight="bold", fontsize=14)
ax1.set_xlabel("Wavenumber k", fontsize=14,fontweight="bold")
    #ax.set_ylabel(r"$\log_{10} E(k)$")
ax1.spines['bottom'].set_linewidth(2.5)
ax1.spines['left'].set_linewidth(2.5)
ax1.spines['top'].set_linewidth(2.5)
ax1.spines['right'].set_linewidth(2.5)
ax1.tick_params(direction='in', length=6, width=2, labelsize=18)



u_slice = u[:, idx_list[2]]
spectrum = np.fft.fftshift(np.abs(np.fft.fft(u_slice)))
# Avoid log(0)
spec_log = np.log10(spectrum + 1e-12)
ax2.plot(k, spec_log, lw=2, color='black')
ax2.set_title(f"{labels[2]}  t = {idx_list[2]/100}", loc='left',fontweight="bold", fontsize=14)
ax2.set_xlabel("Wavenumber k", fontsize=14,fontweight="bold")
    #ax.set_ylabel(r"$\log_{10} E(k)$")
ax2.spines['bottom'].set_linewidth(2.5)
ax2.spines['left'].set_linewidth(2.5)
ax2.spines['top'].set_linewidth(2.5)
ax2.spines['right'].set_linewidth(2.5)
ax2.tick_params(direction='in', length=6, width=2, labelsize=18)



u_slice = u[:, idx_list[3]]
spectrum = np.fft.fftshift(np.abs(np.fft.fft(u_slice)))
# Avoid log(0)
spec_log = np.log10(spectrum + 1e-12)
ax3.plot(k, spec_log, lw=2, color='black')
ax3.set_title(f"{labels[3]}  t = {idx_list[3]/100}", loc='left',fontweight="bold", fontsize=14)
ax3.set_xlabel("Wavenumber k", fontsize=14 ,fontweight="bold")
    #ax.set_ylabel(r"$\log_{10} E(k)$")
ax3.spines['bottom'].set_linewidth(2.5)
ax3.spines['left'].set_linewidth(2.5)
ax3.spines['top'].set_linewidth(2.5)
ax3.spines['right'].set_linewidth(2.5)
ax3.tick_params(direction='in', length=6, width=2, labelsize=18)



ax0.set_ylabel(r"$\log_{10} E(k)$",  fontsize=16)
ax2.set_ylabel(r"$\log_{10} E(k)$", fontsize=16 )



plt.savefig("exp2.eps", bbox_inches='tight', format='eps')
plt.show()

















