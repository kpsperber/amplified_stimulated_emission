import numpy as np
import matplotlib.pyplot as plt

# Defines scaling
nm = 1e-9
us = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

s = 1

# Constants
χ3 = 1.59 * 1e-22 # At 532nm
σ = -1.15
λ = 800 * nm
n0 = 1.4704 # 800nm

β = np.pi / 8

γ0 = 6 * np.pi / (8 * λ * n0)
γ_parallel = γ0 * (1 - (σ / 2) * np.sin(2 * β) ** 2)
γ_perp = -γ0 * (σ / 4) * np.sin(4 * β)
c = 3 * 10 ** 8 * (m / s)

# Model Setup
L = 2 * mm
Nz = 50000
dz = L / Nz

t = np.linspace(-5, 5, 100)

# Initial Conditions
E0 = 0.1 * np.exp(-t ** 2 / 5)
X0 = np.zeros_like(E0)

fig, ax = plt.subplots(1)
ax.plot(t, E0, label = "E-field")
ax.plot(t, X0, label = "XPW")
plt.show()

E = np.zeros((Nz, len(E0)), dtype=complex)
X = np.zeros((Nz, len(X0)), dtype=complex)
E[0] = E0
X[0] = X0


def f(gamma, E):
    return -1j * gamma * E * np.abs(E) ** 2

# Solver
for i in range(0, Nz - 1):
    E[i + 1] = E[i] * np.exp(-1j * γ_parallel * dz * np.abs(E[i]) ** 2)
    X[i + 1] = X[i] -1j * dz * γ_perp * E[i] * np.abs(E[i]) ** 2

fig, ax = plt.subplots(1)
ax.plot(t, np.abs(E[-1]), label = "E-field")
ax.plot(t, np.abs(X[-1]), label = "XPW")
plt.show()

print()