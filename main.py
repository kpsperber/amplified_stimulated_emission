import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift, fftfreq
import tools

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

w_min = 2 * np.pi * c / lamb.max()
w_max = 2 * np.pi * c / lamb.min()

w = np.linspace(w_min, w_max, N)

# Use simulated data
if simulation:
    EX = 0
    EF = 0
    EA = 0

# Read in critical data
else:
    pass

# Generate initial guess

# Update cross polarized and fundamental wave phase and  cross polarized amplitude

# Update fundamental wave amplitude

# Export and plot the results
