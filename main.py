import numpy as np
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

c = (3 * 10 ** 8 ) * m / s

simulation = True
debug = True

N = 2048 # I'm not sure this is the correct resolution
lamb = np.linspace(190, 1100, N) * nm # Range comes from the flame spectrometer and may need to be adjusted
lamb0 = 808 * nm
delta_lamb = 100 * nm
w0 = 2 * np.pi * c / (lamb0)
delta_w = 2 * np.pi * c / delta_lamb

ω_min = 2 * np.pi * c / lamb.max()
ω_max = 2 * np.pi * c / lamb.min()

ω = np.linspace(ω_min, ω_max, N)
dω = ω[1] - ω[0]

# Use simulated data
if simulation:
    lambA0 = 780 * nm
    lambX0 = 808 * nm
    lambF0 = 808 * nm
    delta_lambA = 150 * nm
    delta_lambX = 135 * nm
    delta_lambF = 75 * nm
    τ = 40 * fs

    AF = 1 * np.exp(-(lamb - lambF0) ** 2 / delta_lambF ** 2)
    IF = np.abs(AF) ** 2

    ω_sort, AFω = tools.λ2ω(AF, lamb)
    AFt = tools.ift(AFω)
    AXt, φX = tools.xpw(AFt)
    AXω = tools.ft(AXt)
    _, AX = tools.λ2ω(AXω, ω)

    IX = np.abs(AX) ** 2
    AA = np.sqrt(0.25) * np.exp(-(lamb - lambA0) ** 2 / delta_lambA ** 2)
    IA = np.abs(AA) ** 2

    I = IA + IF + IX + AF * AX * np.exp(1j * (ω * τ)) + AF * AX * np.exp(-1j * (ω * τ)) 

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(lamb * 1e6, IF, label = r"$I_F$")
    ax[0].plot(lamb * 1e6, IA, label = r"$I_A$")
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"I")
    ax[0].legend()

    ax[1].plot(lamb * 1e6, IX, label = r"$I_X$")
    ax[1].plot(lamb * 1e6, IA + IF, label = r"$I_{FA}$")
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"I")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# Read in critical data
else:
    pass

# Generate initial guess
I_hat = tools.ft(I)
t = tools.ω2t(ω)

I_abs = np.abs(I_hat / I_hat.max())

lower_init = 15 * fs
upper_init = 65 * fs

fig, ax = plt.subplots(figsize=(8,4))
plt.subplots_adjust(bottom=0.25)  # Leave space for sliders
(line,) = ax.plot(t, I_abs, lw=1.2)
(vlower,) = ax.plot([lower_init, lower_init], [0, 1], 'r--', lw=1)
(vupper,) = ax.plot([upper_init, upper_init], [0, 1], 'r--', lw=1)
ax.set_xlim(-0.75e-13, 0.75e-13)
ax.set_xlabel("Time delay (s)")
ax.set_ylabel("|Î| (normalized)")
ax.set_title("Interactive Mask Selection")

# --- Slider axes (x-position, y-position, width, height)
ax_lower = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_upper = plt.axes([0.15, 0.06, 0.7, 0.03])

# --- Define sliders
slider_lower = Slider(ax_lower, "Lower (s)", t.min(), t.max(), valinit=lower_init, valstep=fs)
slider_upper = Slider(ax_upper, "Upper (s)", t.min(), t.max(), valinit=upper_init, valstep=fs)

# --- Update function
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

# --- Connect sliders
slider_lower.on_changed(update)
slider_upper.on_changed(update)

plt.show()

lower = slider_lower.val
upper = slider_upper.val
mask = (t >= lower) & (t <= upper)

I_hat_chop = np.zeros_like(I_hat)
I_hat_chop[mask] = I_hat[mask]

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(t, np.abs(I_hat_chop / I_hat.max()))
    ax.set_xlim(-0.75e-13, 0.75e-13)
    plt.show()

# I_hat_chop = tools.recenter(I_hat_chop, t)

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(t, np.abs(I_hat_chop / I_hat.max()))
    ax.set_xlim(-0.75e-13, 0.75e-13)
    plt.show()

I_chop = tools.ift(I_hat_chop)
AFX = np.abs(I_chop)
φ = np.unwrap(np.angle(I_chop))

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(ω, np.abs(I_chop / I_chop.max()))
    plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(ω, φ)
    plt.show()



# Update cross polarized and fundamental wave phase and  cross polarized amplitude

tol = 1e-5
max_iter = 1000
ε_new = np.inf
ε_old = np.inf

AAC = AF_compute = np.abs(I_chop)
φAC = φF_compute = np.unwrap(np.angle(I_chop))



error = []

# plt.ion()
# fig_iter, ax_iter = plt.subplots()
# (line_AAC,) = ax_iter.plot(ω, AAC, label="AAC (target)")
# (line_AF,)  = ax_iter.plot(ω, AF_compute, label="AF_compute")
# ax_iter.set_ylim(0, 1.2 * max(AAC.max(), AF_compute.max()))
# ax_iter.legend()
# ax_iter.set_title("Amplitude Evolution per Iteration")
# ax_iter.set_xlabel("ω (rad/s)")
# ax_iter.set_ylabel("Amplitude")


for iteration_count in range(max_iter):
    α = 1.000 + 0.15 * (iteration_count / max_iter)

    if iteration_count < max_iter * 0.4:
        sigma = 2.0
    elif iteration_count < max_iter * 0.7:
        sigma = 1.0
    else:
        sigma = 0.5

    # --- Phase correction (only where signal is strong)
    EX, φF_compute = tools.phase_correction(
        AF_compute, φF_compute, φAC, ω, np.abs(I_chop)
    )
    AX_compute = np.abs(EX)

    # --- Compute error vs target AC amplitude
    ε_old = ε_new
    ε_new = tools.compute_error(np.sqrt(AX_compute * np.abs(AF_compute)), AAC, ω)

    # --- Amplitude correction (gentle, clipped, masked)
    AF_compute = tools.amplitude_correction(
        AF_compute, AX_compute, np.abs(I_chop),
        α = α
    )

    AF_compute = gaussian_filter1d(AF_compute, sigma=sigma)

    # --- Normalize to avoid growth/decay
    max_val = np.max(np.abs(AF_compute))
    if max_val > 0:
        AF_compute /= max_val
        # AF_compute = AF_compute * AAC.max()



    print(f"Iteration {iteration_count}, error {ε_new}")
    error.append(ε_new)

    # line_AF.set_ydata(AF_compute)
    # line_AAC.set_ydata(AAC)
    # ax_iter.set_ylim(0, 1.2 * max(AAC.max(), AF_compute.max()))
    # fig_iter.canvas.draw()
    # fig_iter.canvas.flush_events()

    if np.isfinite(ε_old) and np.isfinite(ε_new):
        if np.abs(ε_new - ε_old) < tol:
            break




error = np.array(error)
fig, ax = plt.subplots(1)
ax.plot(error, "o")
ax.set_yscale("log")
ax.set_ylabel("Error")
ax.set_xlabel("Iteration")

plt.show()

fig, ax = plt.subplots(1)
ax.plot(ω, AF, label = "Original")
ax.plot(ω, AF_compute, "--", label = "Numerical")
plt.legend()

plt.show()

fig, ax = plt.subplots(1)
ax.plot(ω, φAC - φAC[len(φAC) // 2], label = "Original")
ax.plot(ω, φF_compute - φF_compute[len(φF_compute) // 2], label = "Numerical")
plt.legend()

plt.show()


