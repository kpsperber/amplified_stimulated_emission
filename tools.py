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
from scipy.signal import savgol_filter


# Defines scaling
nm = 1e-9
us = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

# Constants
χ3 = 1.59 * 1e-22 # At 532nm
σ = -1.15
λ = 800 * nm
n0 = 1.4704 # 800nm

β = np.pi / 8

γ0 = 6 * np.pi / (8 * λ * n0)
γ1 = γ0 * (1 - (σ / 2) * np.sin(2 * β) ** 2)
γ2 = -γ0 * (σ / 4) * np.sin(4 * β)

c = 3e8


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

    return epsilon / len(x) ** 2

def ft(A):
    return fftshift(fft(ifftshift(A)))

def ift(A_hat):
    return fftshift(ifft(ifftshift(A_hat)))

def B_integral(I, n, lamb):
    result = 0

    result = 2 * np.pi * result / lamb
    return result

def xpw(A, L = 2 * mm): # Figure out crystal length
    I = np.abs(A) ** 2
    phase = γ1 * I * L
    A_new = A * np.exp(-1j * phase)
    B = A * (γ2 / γ1) * (np.exp(-1j * phase) - 1)

    φX = np.unwrap(np.angle(B))

    return B, φX
    

def compute_xpw(AF, φF):
    EF = ift(AF * np.exp(1j * φF))

    EX, φX = xpw(EF)
    φX = np.unwrap(np.angle(ft(EX)))

    return EF, φX


def compute_rms(φF, φX, φAC, ω):
    φACR = φF - φX
    ε = compute_error(φACR, φAC, x = ω)

    return ε

def improve_f_wave(φAC, φX):
    φF = φAC - φX
    
    return φF

def λ2ω(A, λ):
    ω = 2 * np.pi * c / λ
    idx = np.argsort(ω)
    A_new = A[idx]

    return ω[idx], A_new

def smooth_phase(phi):
    return savgol_filter(phi, 51, 3)

def phase_correction(AF, φF, φAC, ω, IAC, β=0.01):
    EX, φX = compute_xpw(AF, φF)
    error_phase = φAC - φX

    target = np.sqrt(IAC)
    thresh = 0.05 * np.max(target)
    reliable = target > thresh

    φF_new = φF.copy()
    φF_new = φF + β * error_phase

    # ---- Fix A: smooth the phase ----
    φF_new = np.unwrap(smooth_phase(φF_new))

    return EX, φF_new


def amplitude_correction(AF, AX, IAC, α=0.05, eps=1e-12, clip=3.0):
    target = np.sqrt(IAC)
    thresh = 0.05 * np.max(target)
    reliable = target > thresh

    AF_new = AF.copy()

    denom = AX * np.abs(AF) + eps
    correction = (target / denom) ** α

    # clip correction (prevents blowup)
    correction = np.clip(correction, 1.0/clip, clip)

    AF_new[reliable] = AF[reliable] * correction[reliable]

    # Optionally slightly dampen wings:
    AF_new[~reliable] *= 0.95

    return AF_new




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
