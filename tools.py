import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift, fftfreq
from scipy import integrate
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import numpy.fft as nfft
from scipy.interpolate import RegularGridInterpolator

def ft(a):
    """Centered 1D FFT of a (already sampled on a uniform grid)."""
    return nfft.fftshift(nfft.fft(nfft.ifftshift(a)))

def ift(Ahat):
    """Centered 1D inverse FFT."""
    return nfft.fftshift(nfft.ifft(nfft.ifftshift(Ahat)))

def ω2t(omega):
    """
    Given uniform angular-frequency grid omega (rad/s),
    return centered time grid t (s) matching fftshifted spectra.
    """
    N = omega.size
    dω = omega[1] - omega[0]
    # fftfreq expects the sampling interval of the *Hz* grid: d(f) = dω/(2π)
    t = nfft.fftshift(nfft.fftfreq(N, dω / (2*np.pi)))
    return t

def recenter(I_hat, t):
    """Recenters peak of I_hat (complex) to t=0 using interpolation."""
    t_peak = t[np.argmax(np.abs(I_hat))]
    interp = RegularGridInterpolator((t,), I_hat, bounds_error=False, fill_value=0.0)
    # Evaluate at t + t_peak so the original peak appears at 0 on the same x-axis
    return interp(t + t_peak)

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

def polarizer(E_field, polarization = "", theta = None):
    
    if polarization.lower() == "horizontal":
        P = np.array([
            [1, 0],
            [0, 0]
        ])

    elif polarization.lower == "vertical":
        P = np.array([
            [0, 0],
            [0, 1]
        ])

    elif polarization == "" and theta is not None:
        P = np.array([
            [np.cos(theta) ** 2, np.cos(theta) * np.sin(theta)],
            [np.cos(theta) * np.sin(theta), np.sin(theta) ** 2]
        ])

    else:
        P = np.eye(2)

    E_polarized = P @ E_field
    
    return E_polarized

def ft(A):
    return fftshift(fft(ifftshift(A)))

def ift(A_hat):
    return fftshift(ifft(ifftshift(A_hat)))

def B_integral(I, n, lamb):
    result = 0

    result = 2 * np.pi * result / lamb
    return result

def xpw(E, L = None): # Figure out crystal length
    pass

def compute_xpw(AF, φF):
    EF = ift(AF * np.exp(1j * φF))
    φX = np.unwrap(np.angle(ft(xpw(EF))))

    pass

def compute_rms(φF, φX, φAC, ω):
    φACR = φF - φX
    ε = compute_error(φACR, φAC, x = ω)

    return ε

def improve_f_wave(φAC, φX):
    φF = φAC - φX
    
    return φF

def phase_correction(AF_compute, φF_compute, φAC, ω):
    ε_new = np.inf
    ε_old = np.inf
    
    while  ε_new >= ε_old:
        EX, φX = compute_xpw(AF_compute, φF_compute)
        
        ε_old = ε_new
        ε_new = compute_rms(φF_compute, φX, φAC, ω)

        if ε_new > ε_old:
            φF = improve_f_wave(φAC, φX)

    return EX, φF

def amplitude_correction(AX, IAC):
    AF = IAC / AX

    return AF



if __name__ == "__main__":
    m = 1e3
    mm = 1
    μm = 1e-3
    nm = 1e-6

    σx = 1.0 * mm
    σy = 1.0 * mm 
    λ = 808 * nm            # wavelength (arbitrary units)
    zRx = np.pi * σx**2 / λ
    zRy = np.pi * σy**2 / λ

    x = np.linspace(-5 * mm, 5 * mm, 300)
    y = np.linspace(-5 * mm, 5 * mm, 300)
    z = np.linspace(-10, 10, 200)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Gaussian beam intensity
    wx = σx * np.sqrt(1 + (Z / zRx)**2)
    wy = σy * np.sqrt(1 + (Z / zRy) ** 2)
    I = np.exp(-2 * X**2 / wx**2) * np.exp(-2 * Y**2 / wy**2)

    # Take central slices through the beam
    mid_y = len(y) // 2
    mid_x = len(x) // 2

    Ixz = I[:, mid_y, :]  # x–z plane (y fixed)
    Iyz = I[mid_x, :, :]  # y–z plane (x fixed)

    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    im0 = ax[0].pcolormesh(x, z, Ixz.T, shading='auto', cmap='plasma')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('z')
    ax[0].set_title('Gaussian Beam (x–z plane)')
    fig.colorbar(im0, ax=ax[0], label='Intensity')

    im1 = ax[1].pcolormesh(y, z, Iyz.T, shading='auto', cmap='plasma')
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('z')
    ax[1].set_title('Gaussian Beam (y–z plane)')
    fig.colorbar(im1, ax=ax[1], label='Intensity')

    plt.tight_layout()
    plt.show()
