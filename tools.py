import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift, fftfreq
from scipy import integrate
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
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
γ_parallel = γ0 * (1 - (σ / 2) * np.sin(2 * β) ** 2)
γ_perp = -γ0 * (σ / 4) * np.sin(4 * β)
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
   
def xpw(E_t, L=2e-3):
    """
    XPW in holographic-cut BaF2 (plane-wave, undepleted pump model)
    E_t : fundamental field vs time (complex)
    L   : crystal length (m)
    Returns:
        EX_t : XPW field vs time
        EF_t : fundamental after SPM vs time
    """
    I = np.abs(E_t)**2
    phase = γ_parallel * I * L               # nonlinear phase shift of FW

    EF_t = E_t * np.exp(-1j*phase)          # fundamental after SPM
    EX_t = E_t * (γ_perp/γ_parallel) * (np.exp(-1j*phase) - 1.0)  # XPW

    return EX_t, EF_t


# def xpw(A):
#     # idealized XPW polarization
#     return A**3, np.unwrap(np.angle(A**3))

def compute_xpw(AF, φF):
    EF_t = ift(AF * np.exp(1j*φF))   # FW(t)
    EX_t, _ = xpw(EF_t)              # XPW(t)
    EXω = ft(EX_t)
    φX  = np.unwrap(np.angle(EXω))
    return EXω, φX


def compute_rms(φF, φX, φAC, ω):
    φACR = φF - φX
    ε = compute_error(φACR, φAC, x = ω)

    return ε

def compute_error2(retrieved, measured):
    N = len(retrieved)
    error = np.sum((retrieved - measured) ** 2) / N
    return error

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

def phase_correction(AF, φF, φAC, ω, IAC):
    EX, φX = compute_xpw(AF, φF)
    error_phase = (φAC - φAC[len(φAC) // 2]) - (φX - φX[len(φX) // 2])
    N = len(ω)

    ε0 = np.sum(np.abs(φF - φAC) ** 2) / N

    β = min(0.01, 0.5 * ε0)
    β = max(β, 1e-4)

    φF_new = φF.copy()
    φF_new = φF + β * error_phase

    φF_new = np.unwrap(smooth_phase(φF_new))
    ε = np.sum(np.abs((φF_new - φF_new[len(φF_new) // 2]) - (φAC - φAC[len(φAC) // 2])) ** 2) / N

    return EX, φF_new, ε

# def amplitude_correction(AF, AX, IAC, μ=0.2, eps=1e-12):
#     target = np.sqrt(IAC)      # AAC
#     AF_target = target / (AX + eps)

#     # relaxed update: convex combination
#     AF_new = (1 - μ) * AF + μ * AF_target
#     return AF_new


def amplitude_correction(AF, AX, IAC, α=0.3, eps=1e-12, clip=3.0):
    """
    Amplitude update based on AC amplitude:
        target ≈ |AF| * AX

    AF : current fundamental spectrum (complex)
    AX : current XPW amplitude (real, >= 0)
    IAC: measured |AC|^2 (i.e. |I_chop|)
    α  : damping exponent (0 < α <= 1)
    """
    # Measured AC amplitude
    target = np.sqrt(IAC)              # AAC

    # Current model AC amplitude
    product = np.abs(AF) * AX + eps

    # Ratio between measured and model AC
    ratio = target / product           # want product * ratio → target

    # Damped update: AF_new = AF * ratio^α
    corr = ratio ** α

    # Optional clipping for stability
    corr = np.clip(corr, 1.0/clip, clip)

    AF_new = AF * corr

    return AF_new

def read_txt(filepath, as_numpy=False, negs_to_zero=False):
    '''
    Reads a text file from spectrometer into pandas DataFrame or numpy array.

    Args:
        filepath (str): Path to the text file.
        as_numpy (bool): If True, returns data as a 2D numpy array instead of pandas dataframe. Default is False.
        negs_to_zero (bool): If True, negative intensity values are set to zero. Default is False.
    
    Returns:
        pd.DataFrame or np.ndarray: Spectral data as a pandas DataFrame or a 2D numpy array. If a numpy array is
            returned, the first row contains wavelengths and the second row contains intensities.
    '''
    data = pd.read_csv(filepath, delimiter='\t', skiprows=14, names=['wavelength', 'intensity'], index_col='wavelength')
    if negs_to_zero:
        data = data.where(data['intensity'] > 0, 0)
    if as_numpy:
        idx = data.index.to_numpy()
        vals = data['intensity'].to_numpy()
        print('Returning 2D numpy array. a[0,:] are indexes and a[1,:] are values')
        return np.stack((idx, vals), axis=0)
    return data

def centroid_calc_np_1d(data, coord_array):
    '''
    Computes the centroid of the input dataset in 1 dimension through a weighted average sum.

    Args:
        data (np.array): Input data to find the centroid of.
        coord_array (np.array): Array of coordinate values corresponding to the data points.
    
    Returns:
        centroid (np.array): Array of 1 value containing the centroid
    '''
    centroid = np.sum(data * coord_array) / np.sum(data)
    return centroid

def polynomial_peak_fit_np(data, coord_array, window_size=1):
    '''
    Finds the subpixel peak of a 1D dataset by fitting a quadratic polynomial.

    Args:
        data (np.array): Input data to find the peak of.
        coord_array (np.array): Array of coordinate values corresponding to the data points.
        window_size (int): Number of points on either side of the coarse peak to include in the fit. Default is 1.

    Returns:
        fx_peak (float): Subpixel peak position.
    '''

    coarse_peak_idx = np.argmax(data)
    x0 = coord_array[coarse_peak_idx]
    window = data[max(0, coarse_peak_idx - window_size): min(len(data), coarse_peak_idx + window_size + 1)] # Ensure can't go out of bounds
    xs_rel = x0 - coord_array[max(0, coarse_peak_idx - window_size): min(len(data), coarse_peak_idx + window_size + 1)]

    # Fit quadratic: f(x) = ax² + bx + c
    X = np.stack([
        xs_rel.ravel()**2,
        xs_rel.ravel(),
        np.ones_like(xs_rel.ravel())
    ], axis=1)
    
    coeffs, _, _, _ = np.linalg.lstsq(X, window, rcond=None)
    a, b, c = coeffs

    # Subpixel peak from derivative = 0
    if 2*a == 0:
        return x0  # fallback
    dx = b / (2 * a)
    
    if np.iscomplex(dx): dx = dx.real # Enforce real in case complex output

    fx_peak = x0 + dx
    return fx_peak