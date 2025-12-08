import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy.fft import fft, ifft, fftshift, ifftshift
import tools
from matplotlib.widgets import Slider

# Scaling
nm = 1e-9
μm = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

s = 1
fs = 1e-15

c = 3e8 * (m / s)

# ---------- Core functions ----------

def xpw(Aω, φω, k = 1.0):
    """Model XPW field: here we use the analytical A^3 e^{i 3φ} form."""
    Eω = Aω * np.exp(1j * φω)
    # You keep the FFT version for later if you want,
    # but for now we just use the analytic dependence:
    return k * (Aω**3) * np.exp(1j * 3 * φω)

def grid_transform(λ, Aλ):
    ω_min, ω_max = 2 * np.pi * c / λ.max(), 2 * np.pi * c / λ.min()
    N = len(λ)
    ω = np.linspace(ω_min, ω_max, N)
    
    λ_target = 2 * np.pi * c / ω
    F = interp1d(λ, Aλ, kind = "cubic", fill_value = 0.0)

    Aω = F(λ_target)
    return ω, Aω

def remove_const_and_linear(φ, ω, ω0):
    """Subtract best-fit constant + linear trend from φ(ω)."""
    dw = ω - ω0
    a1, a0 = np.polyfit(dw, φ, 1)
    return φ - (a0 + a1 * dw)


def phase(ω, ω0, params, fs = 1e-15):
    """
    Polynomial phase: φ(ω) = sum_i params[i] * fs^i * (ω - ω0)^i
    params[0] = constant, params[1] = linear, params[2] = quadratic, ...
    """
    φ = np.zeros_like(ω)
    for i in range(len(params)):
        φ = φ + params[i] * fs**i * (ω - ω0)**i
    return φ

def amplitude_gauss(ω, ω0, log_sigma, A0_ref):
    """
    Gaussian amplitude with fixed peak A0_ref and width given by log_sigma.
    params here is just a scalar log_sigma.
    """
    σ = np.exp(log_sigma)
    return A0_ref * np.exp(-(ω - ω0)**2 / (2 * σ**2))

# ---------- Helpers for initial guess ----------

def gaussian_moment_guess(A, ω):
    """
    Estimate Gaussian width from moments of |A(ω)|^2.
    Returns (log_A0_est, log_sigma_est), but we will only use log_sigma_est.
    """
    A_norm = A / (np.max(A) + 1e-20)
    w = A_norm**2
    w[w < 1e-6] = 0.0

    W = np.sum(w) + 1e-20
    μ = np.sum(w * ω) / W
    var = np.sum(w * (ω - μ)**2) / W
    σ_est = np.sqrt(var)

    A0_est = np.max(A)
    return np.log(A0_est), np.log(σ_est)

def phase_poly_guess(φ, ω, ω0, fs=1e-15, order=3):
    """
    Fit φ(ω) ≈ Σ_i params[i] * (fs * (ω - ω0))^i in a least-squares sense.
    The returned params can be used directly in `phase(ω, ω0, params, fs)`.
    """
    dω = ω - ω0
    u = fs * dω

    X = np.vstack([u**i for i in range(order)]).T

    coeffs, *_ = np.linalg.lstsq(X, φ, rcond=None)
    phase_params = coeffs

    return phase_params


def initial_guess(A, φ, ω, ω0, order=3):
    """
    Build initial parameter vector:
        params = [log_sigma, φ0, φ1, φ2]  (length = 1 + order)
    Amplitude peak A0_ref is handled separately outside.
    """
    params0 = np.zeros(1 + order)

    _, log_sigma_est = gaussian_moment_guess(A, ω)
    params0[0] = log_sigma_est

    fig, ax = plt.subplots(1)
    ax.plot(ω, φ)

    plt.show()
    phase_params = phase_poly_guess(φ, ω, ω0, order=order)
    params0[1:] = phase_params

    return params0

# ---------- Forward model & cost ----------

def forward(ω, ω0, params, τ_delay, A0_ref, AA = None, scale = 0.3):
    """
    params = [log_sigma, φ0, φ1, φ2]
    """
    log_sigma = params[0]
    phase_params = params[1:]

    Aω = amplitude_gauss(ω, ω0, log_sigma, A0_ref)
    φω = phase(ω, ω0, phase_params)

    EFω = Aω * np.exp(1j * φω)
    EXω = xpw(Aω, φω)  # XPW field
    EXω = EXω * np.exp(1j * ω * τ_delay)

    if AA is not None:
        I = np.abs(EFω + EXω)**2 + np.abs(AA) ** 2
    else:
        I = np.abs(EFω + EXω)**2

    return I

def cost_function(params, ω, ω0, τ_delay, A0_ref, I_measured, φ_prior=None, w_I=1.0, w_φ=1e-3):

    I_sim = forward(ω, ω0, params, τ_delay, A0_ref)
    
    I_sim_normalize = I_sim / (np.max(I_sim) + 1e-20)
    I_measured_normalize = I_measured / np.max(I_measured)

    num = np.dot(I_measured_normalize, I_sim_normalize)
    denom = np.dot(I_sim_normalize, I_sim_normalize) + 1e-20
    α = num / denom

    resid = I_measured_normalize - α * I_sim_normalize
    C_I = np.sum(resid**2)

    # Phase-prior term (optional)
    C_φ = 0.0
    if φ_prior is not None:
        log_sigma = params[0]
        phase_params = params[1:]
        φ_model = phase(ω, ω0, phase_params)

        # match the same gauge as φ_prior
        C_φ = np.sum((φ_model - φ_prior)**2) / len(ω)

    return w_I * C_I + w_φ * C_φ



# ---------- Run simulation ----------

simulation = True
debug = True

if simulation:
    # Build grids
    N = 2048
    λ_min = 190 * nm
    λ_max = 1100 * nm
    λ = np.linspace(λ_min, λ_max, N)

    # Beam characteristics
    λ0 = 808 * nm
    Δλ = 75 * nm

    ω0 = 2 * np.pi * c / λ0
    Δω = 2 * np.pi * c / Δλ

    τ = 80 * fs
    τ_delay = 100 * fs
    
    AF = np.exp(-(λ - λ0)**2 / Δλ**2)
    AA = 0.075 * np.exp(-(λ - λ0)**2 / Δλ**2)
    ω, AFω = grid_transform(λ, AF)
    _, AAω = grid_transform(λ, AA)
    dω = ω[1] - ω[0]
    dt = 2 * np.pi / (N * dω)
    t = (np.arange(N) - N // 2) * dt

    # True parameters
    sigma_true = (2 * np.pi * c / (λ0**2)) * Δλ
    log_sigma_true = np.log(sigma_true)
    true_phase_params = np.array([1, 20, 15])   # φ0, φ1, φ2
    true_params = np.concatenate([[log_sigma_true], true_phase_params])

    A0_ref_true = AFω.max()
    I = forward(ω, ω0, true_params, τ_delay, A0_ref_true, AA=AAω)

    # Quick check of simulated interferogram
    fig, ax = plt.subplots(1)
    ax.plot(ω, I / I.max())
    ax.set_title("Simulated I(ω)")
    plt.show()

else:
    pass


# Extract AC peak
I_hat = tools.ft(I)
t = tools.ω2t(ω)
low = 60 * fs
high = 800 * fs

I_abs = np.abs(I_hat / I_hat.max())
fig, ax = plt.subplots(1)
ax.plot(t, I_abs)
ax.vlines([low, high], ymin=0, ymax=1.0, color = "red")
ax.set_title("|I(t)|")
plt.show()

lower = low
upper = high
mask = (t >= lower) & (t <= upper)

I_hat_chop = np.zeros_like(I_hat)
I_hat_chop[mask] = I_hat[mask]
I_hat_chop = tools.recenter(I_hat_chop, t)

I_chop = tools.ift(I_hat_chop)
AFω0 = np.abs(I_chop)
φ0 = np.unwrap(np.angle(I_chop))

ids = AFω0 > 0.05 * AFω0.max()
ω = ω[ids]
AFω0 = AFω0[ids]
φ0 = φ0[ids] + ω * τ_delay
N_sample = len(φ0)
φ0 = φ0 - φ0[N_sample // 2]

if simulation:
    I = I[ids]

# Reference amplitude scale for retrieval
A0_ref = AFω0.max()

# Forms initial guess
x0 = initial_guess(AFω0, φ0, ω, ω0, order=3)

if debug:
    fig, ax = plt.subplots(1, 2)
    fig.suptitle("Initial Guess")

    # Compare amplitude shape
    ax[0].plot(ω, amplitude_gauss(ω, ω0, x0[0], A0_ref), label="amp guess")
    if simulation:
        ax[0].plot(ω, AFω0, "--", label="chop amp")
    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()

    # Compare phase
    ax[1].plot(ω, φ0, label = "Measured")
    ax[1].plot(ω, phase(ω, ω0, x0[1:]), label="phase guess")
    if simulation:
        φ_true = phase(ω, ω0, true_phase_params)
        ax[1].plot(ω, φ_true, "--", label="true phase")
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"$\phi(\omega)$")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# ---------- Final optimization ----------

res = minimize(
    cost_function,
    x0,
    args=(ω, ω0, τ_delay, A0_ref, I, φ0, 1.0, 0),
    method="Nelder-Mead", 
    options={"maxiter": 1000, "disp": True}
)

x = res.x
log_sigma_fit = x[0]
phase_params_fit = x[1:]
x0 = x

# --- Rebuild coherent fields from fit and extract ASE ---

if simulation:
    Aω_true = amplitude_gauss(ω, ω0, log_sigma_true, A0_ref_true)
    φ_true  = phase(ω, ω0, true_phase_params, fs=fs)
    EFω_true = Aω_true * np.exp(1j * φ_true)
    EXω_true = xpw(Aω_true, φ_true) * np.exp(1j * ω * τ_delay)

Aω_fit = amplitude_gauss(ω, ω0, log_sigma_fit, A0_ref)
φ_fit  = phase(ω, ω0, phase_params_fit, fs=fs)

EFω_fit = Aω_fit * np.exp(1j * φ_fit)
EXω_fit = xpw(Aω_fit, φ_fit) * np.exp(1j * ω * τ_delay)

I_coh = np.abs(EFω_fit + EXω_fit)**2

num   = np.dot(I, I_coh)
denom = np.dot(I_coh, I_coh) + 1e-20
α_coh = num / denom

I_coh_scaled = α_coh * I_coh
I_ASE = I - I_coh
A_ASE = np.sqrt(I_ASE)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Coherent vs ASE decomposition")

ax[0].plot(ω, I, label="Measured I(ω)")
ax[0].plot(ω, I_coh_scaled, "--", label="Coherent model")
ax[0].set_title("Total vs coherent-only")
ax[0].legend()


ax[1].plot(ω, np.abs(I_ASE), label = "Measured")

if simulation:
    ax[1].plot(ω, np.abs(AAω[ids]) ** 2, label = "Real")

ax[1].set_title("ASE Intensity")
ax[1].set_xlabel(r"$\omega$")
ax[1].legend()

plt.tight_layout()
plt.show()

print()
