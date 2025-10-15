import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift, fftfreq
from scipy import integrate
from scipy.integrate import simpson

# Computes the L1 norm over the retrieved and measured data
def compute_error(retrieved, measured, x, y = None):
    diff = np.abs(retrieved - measured)

    if diff.ndim == 0:
        return 0.0
    
    elif diff.ndim == 1 and y is None:
        epsilon = np.trapezoid(diff, x)

    else:
        epsilon = simpson(diff, x = x, axis = -1)
        epsilon = simpson(epsilon, x = y, axis = -1)

    return epsilon