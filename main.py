#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.fft import fftfreq
import tools

#%%
# Initialize scale of our setup
nm = 1e-6
us = 1e-3
mm = 1
cm = 10
m = 1e3

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

#%%
# Use simulated data
if simulation:
    lambA0 = 780 * nm
    lambX0 = 808 * nm
    lambF0 = 808 * nm
    delta_lambA = 150 * nm
    delta_lambX = 135 * nm
    delta_lambF = 75 * nm
    τ = 40 * fs


    AX = np.sqrt(0.6) * np.exp(-(lamb - lambX0) ** 2 / delta_lambX ** 2)
    IX = np.abs(AX) ** 2
    AF = np.exp(-(lamb - lambF0) ** 2 / delta_lambF ** 2)
    IF = np.abs(AF) ** 2
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

#%%
# Generate initial guess
I_hat = tools.ft(I)
t = tools.ω2t(ω)

I_abs = np.abs(I_hat / I_hat.max())

lower_init = 25 * fs
upper_init = 55 * fs

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

I_hat_chop = tools.recenter(I_hat_chop, t)

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(t, np.abs(I_hat_chop / I_hat.max()))
    ax.set_xlim(-0.75e-13, 0.75e-13)
    plt.show()

#%%
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

ε_new = np.inf
ε_old = np.inf

AAC = AF_compute = np.abs(I_chop)
φAC = φF_compute = np.unwrap(np.angle(I_chop))

iteration_counter = 0

while ε_new >= ε_old:
    if iteration_counter % 100 == 0:
        print(f"Iteration {iteration_counter}")
        
    EX, φX = tools.phase_correction(AF_compute, φF_compute, φAC, ω)
    AX_compute = np.abs(EX)

    ε_old = ε_new
    ε_new = tools.compute_error(np.sqrt(AX_compute * AF_compute), AAC, ω)

    if ε_new > ε_old:
        tools.amplitude_correction(AX_compute, I_chop)
        iteration_counter = iteration_counter + 1

    else:
        break


# %%
# Export and plot the results



