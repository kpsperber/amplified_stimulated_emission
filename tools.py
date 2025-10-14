import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift, fftfreq

# Computes the L1 norm over the retrieved and measured data
def compute_error(retrieved, measured, grid):
    diff = np.abs(retrieved - measured)
    epsilon = np.trapezoid(diff, x = grid)

    return epsilon