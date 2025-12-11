import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
N_t = 4096
dt = 0.01
b0 = 1.0
omega_c = 1.0

# spatial grid
Nx = 200
x = np.linspace(-10, 10, Nx)

q = 5.0

# RANDOM VARIABLES
mu_f0, sigma_f0 = 0.0, 1.0
f0 = np.random.normal(mu_f0, sigma_f0)


nu = np.random.uniform(0,2*np.pi)
# GAUSSIAN-CORRELATED NOISE ξ(t)
N = N_t
freqs = np.fft.fftfreq(N, d=dt)
omega = 2*np.pi*freqs

Bw = b0 * np.exp(-(omega/omega_c)**2)

dW = np.random.normal(size=N) + 1j*np.random.normal(size=N)
spectrum = np.sqrt(Bw) * dW

xi = np.real(np.fft.ifft(spectrum))
t = np.arange(N) * dt

# SPATIAL FUNCTION f(x) WITH RANDOM f0 AND ν
f_x = f0 * np.cos(q*x + nu)

# FULL STOCHASTIC FIELD
F = np.outer(xi, f_x)


# PLOTS

plt.figure(figsize=(12,4))
plt.plot(x, f_x, lw=2)
plt.title(f"Spatial Function f(x) with random f0={f0:.3f},  nu={nu:.3f}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(t, xi)
plt.title("Gaussian-Correlated Time Noise ξ(t)")
plt.xlabel("t")
plt.ylabel("ξ(t)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.imshow(F, aspect='auto', extent=[x[0], x[-1], t[-1], t[0]], cmap='coolwarm')
plt.colorbar(label="F(x,t)")
plt.title("Full Stochastic Field F(x,t) = f(x) ξ(t)")
plt.xlabel("x")
plt.ylabel("t")
plt.show()
