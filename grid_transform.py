import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

c = 3 * 10 ** 8

λ = np.linspace(100, 1500, 1000) * 1e-9
ω_min, ω_max = 2 * np.pi * c / λ.min(), 2 * np.pi * c / λ.max()
ω = np.linspace(ω_min, ω_max, 1000)

f = np.sin(2 * np.pi * λ / λ.max())

λ_target = 2 * np.pi * c / ω
F = interp1d(λ, f, kind = "cubic", fill_value = 0.0)
f_ω = F(λ_target)


fig, ax = plt.subplots(1)
ax.plot(λ, f)
plt.show()

a = np.sin(4 * np.pi ** 2 * c / (λ.max() * ω))

fig, ax = plt.subplots(1)
ax.plot(ω, a, label = "solution")
ax.plot(ω, f_ω, label = "Interpolation")
ax.legend()
plt.show()
