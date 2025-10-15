#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift, fftfreq
import tools

#%%
# Initialize scale of our setup
nm = 1e-6
us = 1e-3
mm = 1
cm = 10
m = 1e3

fs = 1e-15 # May need to rescale time
s = 1

c = (3 * 10 ** 8 ) * m / s

simulation = True

N = 2048 # I'm not sure this is the correct resolution
lamb = np.linspace(190, 1100, N) * nm # Range comes from the flame spectrometer and may need to be adjusted
lamb_0 = 808 * nm
delta_lamb = 10 * nm
w0 = 2 * np.pi * c / (lamb_0)
delta_w = 2 * np.pi * c / delta_lamb

w_min = 2 * np.pi * c / lamb.max()
w_max = 2 * np.pi * c / lamb.min()

w = np.linspace(w_min, w_max, N)
dw = w[1] - w[0]

#%%
# Use simulated data
if simulation:
    AX = np.exp(-(w - w0) ** 2 / delta_w ** 2)
    AF = np.exp(-(w - w0) ** 2 / delta_w ** 2)
    AA = np.exp(-(w - w0) ** 2 / delta_w ** 2)
    
    EX = 0
    EF = 0
    EA = 0

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(w, AF)
    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel(r"\|E\|")

    ax[1].plot(w, AX)
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"\|E\|")

    ax[2].plot(w, AA)
    ax[2].set_xlabel(r"$\omega$")
    ax[2].set_ylabel(r"\|E\|")

    #%%

    x = np.linspace(0, 1)
    y = np.linspace(0, 1)

    X, Y = np.meshgrid(x, y)
    measure = X ** 2
    retrieved = X ** 2 + 1

    print(f"{tools.compute_error(retrieved, measure, x, grid_2 = y)}")

# Read in critical data
else:
    pass

# Generate initial guess

# Update cross polarized and fundamental wave phase and  cross polarized amplitude

# Update fundamental wave amplitude

# Export and plot the results
