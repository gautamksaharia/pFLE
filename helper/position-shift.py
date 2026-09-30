import numpy as np


def gamma(t, gamma0, Gamma):
    return gamma0 / (1.0 + 2.0 * Gamma * gamma0 * t)


def Delta(t, Delta0, gamma0, Gamma):
    g = gamma(t, gamma0, Gamma)
    return Delta0 * np.exp(Gamma * t) * np.sqrt(
        np.sin(g) / np.sin(gamma0))


def delta_x1_x2(t, Delta10, Delta20, gamma10, gamma20, Gamma):
    # gamma_1(t), gamma_2(t)
    g1 = gamma(t, gamma10, Gamma)
    g2 = gamma(t, gamma20, Gamma)
    # Delta_1(t), Delta_2(t)
    D1 = Delta(t, Delta10, gamma10, Gamma)
    D2 = Delta(t, Delta20, gamma20, Gamma)
    # Numerator and denominator inside the logarithm
    numerator = (D1**4+ D2**4- 2.0 * D1**2 * D2**2 * np.cos(g1 + g2))
    denominator = (D1**4+ D2**4- 2.0 * D1**2 * D2**2 * np.cos(g2 - g1))
    # Delta x_1(t)
    dx1 = (1.0 / (2.0 * D1**2 * np.sin(g1))* np.log(numerator / denominator))
    dx2 = (-1.0 / (2.0 * D2**2 * np.sin(g2))* np.log(numerator / denominator))
    return dx1, dx2



# Parameters
Delta10 = 1.4
Delta20 = 0.9
gamma10 = 0.5*np.pi
gamma20 = 0.5*np.pi
Gamma = 0.00

# Time
t = 15.0

# Calculate Delta x_1
dx1, dx2 = delta_x1_x2(t, Delta10, Delta20, gamma10, gamma20, Gamma)

print("Position shift x_1(t) =", dx1)
print("Position shift x_2(t) =", dx2)
