import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.fft import fftfreq
import tools
from scipy.ndimage import gaussian_filter1d

# Initialize scale of our setup
nm = 1e-9
us = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

x_hat = np.array([1, 0])
y_hat = np.array([0, 1])

fs = 1e-15 # May need to rescale time
s = 1
L = 2 * mm

c = (3 * 10 ** 8 ) * m / s

simulation = True
debug = True

# Use simulated data
if simulation:
    N = 2048
    lamb = np.linspace(190, 1100, N) * nm
    lamb0 = 808 * nm
    delta_lamb = 100 * nm
    w0 = 2 * np.pi * c / (lamb0)
    delta_w = 2 * np.pi * c / delta_lamb

    ω_min = 2 * np.pi * c / lamb.max()
    ω_max = 2 * np.pi * c / lamb.min()

    ω = np.linspace(ω_min, ω_max, N)
    dω = ω[1] - ω[0]

    lambA0 = 780 * nm
    lambX0 = 808 * nm
    lambF0 = 808 * nm
    delta_lambA = 150 * nm
    delta_lambX = 135 * nm
    delta_lambF = 75 * nm
    τ = 40 * fs

    AF = 1 * np.exp(-(lamb - lambF0) ** 2 / delta_lambF ** 2)

    ω_sort, AFω = tools.λ2ω(AF, lamb)
    AFt = tools.ift(AFω)                  # FW(t) at crystal input
    I_t = np.abs(AFt)**2

    # --- Scale field so that nonlinear phase is reasonable (see next section) ---
    eta = np.max(np.abs(tools.γ_parallel * I_t * L))   # current max nonlinear phase
    print("raw max nonlinear phase (γ_parallel * I_max * L) =", eta)

    target_eta = 3.0      # aim for ~3 rad max nonlinear phase in simulation
    scale = np.sqrt(target_eta / eta)   # because I ∝ |E|^2
    AFt *= scale

    # Recompute intensity and spectrum after scaling
    I_t = np.abs(AFt)**2
    AFω = tools.ft(AFt)

    # --- Generate XPW and SPM-modified FW consistently with tools.xpw ---
    AXt, EF_t = tools.xpw(AFt)   # XPW(t), Fundamental after SPM(t)

    EFω = tools.ft(EF_t)
    _, AF = tools.λ2ω(EFω, ω)    # FW spectrum after SPM
    IF = np.abs(AF)**2

    AXω = tools.ft(AXt)
    _, AX = tools.λ2ω(AXω, ω)
    IX = np.abs(AX)**2

    # ASE term
    AA = np.sqrt(0.0025) * np.exp(-(lamb - lambA0) ** 2 / delta_lambA ** 2)
    IA = np.abs(AA)**2

    # Synthetic SRSI interferogram
    I = IA + IF + IX + AF * AX * np.exp(1j * (ω * τ)) + AF * AX * np.exp(-1j * (ω * τ))

    # For debugging, print rescaled nonlinear phase
    eta_rescaled = np.max(np.abs(tools.γ_parallel * I_t * L))
    print("rescaled max nonlinear phase =", eta_rescaled)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(lamb * 1e9, IF, label = r"$I_F$")
    ax[0].plot(lamb * 1e9, IA, label = r"$I_A$")
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"I")
    ax[0].legend()

    ax[1].plot(lamb * 1e9, IX, label = r"$I_X$")
    ax[1].plot(lamb * 1e9, IA + IF, label = r"$I_{FA}$")
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"I")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(lamb * 1e9, I)    
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"I")

    plt.show()

    I_t = np.abs(AFt)**2
    eta = np.max(np.abs(tools.γ_parallel * I_t * L))   # dimensionless phase shift
    print("max nonlinear phase (γ_parallel * I_max * L) =", eta)

# Read in critical data
else:
    xpw_df = pd.read_excel("")
    reference_df = pd.read_excel("fringes_no_xpw")
    combined_df = pd.read_excel("fringes_xpw.xlsx")
    dark_df = pd.read_excel("dark.xlsx")

    lamb = xpw_df["Wavelength"].to_numpy() * nm
    IX = xpw_df["Intensity"].to_numpy()
    IF = reference_df["Intensity"].to_numpy()
    I = combined_df["Intensity"].to_numpy()
    I_dark1 = dark_df["Intensity 1"].to_numpy()
    I_dark2 = dark_df["Intensity 2"].to_numpy()
    I_dark3 = dark_df["Intensity 3"].to_numpy()
    I_dark4 = dark_df["Intensity 4"].to_numpy()
    I_dark5 = dark_df["Intensity 5"].to_numpy()

    I_dark = I_dark1 + I_dark2 + I_dark3 + I_dark4 + I_dark5
    I_dark = I_dark / 5

    IX = IX - I_dark
    IF = IF - I_dark
    I = I - I_dark

    fig, ax = plt.subplots(1)
    ax.plot(lamb / nm, IX, label = "XPW")
    ax.plot(lamb / nm, IF, label = "Fundamental")
    ax.plot(lamb / nm, I, label = "Measured")

    ax.legend()
    plt.show()

    mask = (IX < 0)
    IX[mask] = 0

    mask = (IF < 0)
    IF[mask] = 0

    mask = (I < 0)
    I[mask] = 0

    ω = 2 * np.pi * c / lamb
    AF = np.sqrt(IF)
    AX = np.sqrt(IX)

# Generate initial guess
I_hat = tools.ft(I)
t = tools.ω2t(ω)

I_abs = np.abs(I_hat / I_hat.max())

lower_init = 15 * fs
upper_init = 65 * fs

fig, ax = plt.subplots(figsize=(8,4))
plt.subplots_adjust(bottom=0.25)
(line,) = ax.plot(t, I_abs, lw=1.2)
(vlower,) = ax.plot([lower_init, lower_init], [0, 1], 'r--', lw=1)
(vupper,) = ax.plot([upper_init, upper_init], [0, 1], 'r--', lw=1)
ax.set_xlabel("t")
ax.set_ylabel("|I| (a.u.)")
ax.set_title("Intensity")

ax_lower = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_upper = plt.axes([0.15, 0.06, 0.7, 0.03])

slider_lower = Slider(ax_lower, "Lower (s)", t.min(), t.max(), valinit=lower_init, valstep=fs)
slider_upper = Slider(ax_upper, "Upper (s)", t.min(), t.max(), valinit=upper_init, valstep=fs)

#Update function
def update(val):
    lower = slider_lower.val
    upper = slider_upper.val
    mask = (t >= lower) & (t <= upper)
    I_hat_chop = np.zeros_like(I_hat)
    I_hat_chop[mask] = I_hat[mask]

    # update vertical lines
    vlower.set_xdata([lower, lower])
    vupper.set_xdata([upper, upper])

    # optionally display masked signal overlay
    ax.collections.clear()
    ax.fill_between(t, 0, I_abs, where=mask, color='orange', alpha=0.3)

    fig.canvas.draw_idle()

slider_lower.on_changed(update)
slider_upper.on_changed(update)

plt.show()

lower = slider_lower.val
upper = slider_upper.val
mask = (t >= lower) & (t <= upper)

I_hat_chop = np.zeros_like(I_hat)
I_hat_chop[mask] = I_hat[mask]

I_hat_chop = tools.recenter(I_hat_chop, t)

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(t, np.abs(I_hat_chop / I_hat.max()))
    ax.set_xlim(-0.75e-13, 0.75e-13)
    plt.show()

I_chop = tools.ift(I_hat_chop)
AFX = np.abs(I_chop)
φ = np.angle(I_chop)

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(ω, np.abs(I_chop / I_chop.max()))
    plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(ω, φ)
    plt.show()

# Iteration parameters
tol = 1e-5
phase_tol = 1e-3
max_iter = 1000
ε_amplitude_new = np.inf
ε_amplitude_old = np.inf
ε_phase_new = np.inf
ε_phase_old = np.inf

AAC = np.sqrt(np.abs(I_chop))
φAC = np.unwrap(np.angle(I_chop))

IAC_field = I_chop

# AC amplitude and intensity
AAC = np.abs(IAC_field)
IAC = AAC**2

φAC = np.unwrap(np.angle(IAC_field))
ids = AAC > 0.05 * AAC.max()

IAC_field = IAC_field[ids]
AAC = AAC[ids]
IAC = IAC[ids]
φAC = φAC[ids]
AF = AF[ids]
ω = ω[ids]
AX = AX[ids]
IF = IF[ids]
I = I[ids]
IA = IA[ids]
IX = IX[ids]

# Initial guess
AF_compute = np.abs(IAC) ** 0.5
φF_compute = np.zeros_like(ω)


error_amplitude = []
error_phase = []

plt.ion()
fig_iter, ax_iter = plt.subplots(1, 2)
(line_AF,) = ax_iter[0].plot(ω, AF, label="Original")
(line_AF_compute,)  = ax_iter[0].plot(ω, AF_compute, label="Numerical")
(line_φF,) = ax_iter[1].plot(ω, φAC - φAC[len(φAC) // 2], label="Original")
(line_φF_compute,)  = ax_iter[1].plot(ω, φF_compute - φF_compute[len(φF_compute) // 2], label="Numerical")
ax_iter[0].set_ylim(0, 1.2 * max(AF.max(), AF_compute.max()))
ax_iter[0].legend()
ax_iter[0].set_title("Fundamental Amplitude")
ax_iter[0].set_xlabel("ω (rad/s)")
ax_iter[0].set_ylabel("Fundamental Phase")
ax_iter[1].legend()
ax_iter[1].set_title("Phase")
ax_iter[1].set_xlabel("ω (rad/s)")
ax_iter[1].set_ylabel(r"$\phi$")
ax_iter[1].set_ylim(min((φF_compute - φF_compute[len(φF_compute) // 2]).min(), (φAC - φAC[len(φAC) // 2]).min()), 1.2 * max((φF_compute - φF_compute[len(φF_compute) // 2]).max(), (φAC - φAC[len(φAC) // 2]).max()))

for iteration_count in range(max_iter):

    # Learning Rate
    phase_weight = np.exp(-iteration_count / 150)
    amp_weight   = 1.0 - phase_weight

    EX, φF_compute, ε_phase_new = tools.phase_correction(
        AF_compute, φF_compute, φAC, ω, np.abs(IAC_field)
    )

    # XPW spectrum
    AX_compute = np.abs(EX)

    ε_amplitude_old = ε_amplitude_new
    model_AAC = np.abs(AF_compute) * AX_compute
    ε_amplitude_new = tools.compute_error2(model_AAC, AAC)

    if iteration_count > 10:
        AF_compute = tools.amplitude_correction(
            AF_compute, AX_compute, IAC,
            α = amp_weight * 0.05
        )

    max_val = np.max(np.abs(AF_compute))
    if max_val > 0:
        AF_compute /= max_val

    print(f"Iteration {iteration_count:4d} | Amp Err: {ε_amplitude_new:.4e} | Phase Err: {ε_phase_new:.4e}")

    error_amplitude.append(ε_amplitude_new)
    error_phase.append(ε_phase_new)

    line_AF_compute.set_ydata(AF_compute)
    line_φF_compute.set_ydata(φF_compute - φF_compute[len(φF_compute)//2])

    fig_iter.canvas.draw()
    fig_iter.canvas.flush_events()

error_amplitude = np.array(error_amplitude)
error_phase = np.array(error_phase)
fig, ax = plt.subplots(1, 2)
ax[0].plot(error_amplitude, "o")
ax[0].set_yscale("log")
ax[0].set_ylabel("Amplitude Error")
ax[0].set_xlabel("Iteration")
ax[1].plot(error_phase, "o")
ax[1].set_yscale("log")
ax[1].set_ylabel("Phase Error")
ax[1].set_xlabel("Iteration")

plt.show()

fig, ax = plt.subplots(1)
ax.plot(ω, AF, label = "Original")
ax.plot(ω, AF_compute, "--", label = "Numerical")
plt.legend()

plt.show()

fig, ax = plt.subplots(1)
ax.plot(ω, φAC - φAC[len(φAC) // 2], label = "Original")
ax.plot(ω, φF_compute - φF_compute[len(φF_compute) // 2], "--", label = "Numerical")
plt.legend()

plt.show()

AF_compute = AF_compute
EF_compute = AF_compute * np.exp(1j * φF_compute)
IF_compute = np.abs(EF_compute)**2
IX_known   = np.abs(AX)**2        
AC_compute = EF_compute * AX * np.exp(1j * ω * τ)
I_AC_total = 2 * np.real(AC_compute)
I_compute = IF + IX_known + I_AC_total

fig, ax = plt.subplots(1)
ax.plot(ω, np.abs(I / I.max()), label = "Original")
ax.plot(ω, I_compute / I_compute.max(), "--", label = "Numerical")
plt.legend()

plt.show()

# Extract ASE
IA_retrieved = I - (IF_compute + IX_known + I_AC_total)

A_ASE = np.abs(IA_retrieved)

fig, ax = plt.subplots(1)


if simulation:
    IA = I - (IX + IF + AF * AX * np.exp(1j * (ω * τ)) + AF * AX * np.exp(-1j * (ω * τ)))
    ax.plot(ω, IA, label = "Original")
    ax.plot(ω, IA_retrieved, "--", label = "Numerical")

else:
    ax.plot(ω, IA_retrieved)

plt.legend()
plt.show()

print()


