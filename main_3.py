import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
from numpy.fft import fft, ifft, fftshift, ifftshift
import tools
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter1d
import os

# Scaling
nm = 1e-9
μm = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

s = 1
μs = 1e-6
fs = 1e-15

c = 3e8 * (m / s)

simulation = False

df = pd.read_excel("dark.xlsx")
λ = df["Wavelength"].to_numpy() * nm
N = len(λ)

if simulation:

    # Beam characteristics
    λ0 = 808 * nm
    Δλ = 75 * nm

    ω0 = 2 * np.pi * c / λ0
    Δω = 2 * np.pi * c / Δλ

    τ_delay = 100 * fs
    
    AF = np.exp(-(λ - λ0)**2 / Δλ**2)
    AA = 0.50 * AF
    ω, AFω = tools.grid_transform(λ, AF)
    _, AAω = tools.grid_transform(λ, AA)
    dω = ω[1] - ω[0]
    dt = 2 * np.pi / (N * dω)
    t = (np.arange(N) - N // 2) * dt
    dω = ω[1] - ω[0]
    ω_fft = (np.arange(N) - N // 2) * dω   # FFT-conjugate grid
    φ0 = np.zeros_like(ω)

    EFω_true = AFω * np.exp(1j * φ0)
    EF  = tools.ift(EFω_true)
    EX_true  = EF * np.abs(EF)**2
    EXω_true = 25.0 * tools.ft(EX_true)

    I = np.abs(EFω_true + EXω_true * np.exp(1j * ω * τ_delay)) ** 2 + AAω ** 2
    I_noise = I + 1e-1 * (np.random.rand(I.size) - 0.5)

    ω0 = tools.centroid_calc_np_1d(I_noise, ω)
    gauss = np.exp(-((ω - ω0)/(2 * 2e14))**10)

    fig, ax = plt.subplots(1)
    ax.plot(ω, I_noise / I_noise.max())
    ax.plot(ω, gauss)

    plt.show()


    I_noise = I_noise * gauss

    IAF = np.abs(EFω_true) ** 2 + np.abs(AAω) ** 2

    fig, ax = plt.subplots(1)
    ax.plot(ω, I_noise / I_noise.max())
    ax.set_title("Simulated I(ω)")
    plt.show()

    AXω_known = np.abs(EXω_true)
    

else:
    # Read in files
    xpw_bulk_dir = r"data\xpw_fringes\2025-12-03\callibration_images\xpw_bulk"
    measurement_bulk_dir = r"data\xpw_fringes\2025-12-03\micrometer_637_bulk"
    dark_bulk_dir = r"data\xpw_fringes\2025-12-03\callibration_images\dark_bulk"
    ref_bulk_dir = r"data\xpw_fringes\2025-12-03\callibration_images\reference_arm_bulk"

    z = (float(measurement_bulk_dir.split("\\")[-1].split("_")[1]) * 1e-2) * mm
    τ_delay = 2 * z / c

    xpw_data = np.zeros_like(λ)
    count = 0
    for file in os.listdir(xpw_bulk_dir):
        data = tools.read_txt(xpw_bulk_dir + "\\" + file)
        
        IX = data["intensity"]
        IX[IX < 0] = 0.0
        xpw_data = xpw_data + IX.to_numpy()

        count = count + 1

    xpw_data = xpw_data / count

    I = np.zeros_like(λ)
    count = 0
    for file in os.listdir(measurement_bulk_dir):
        data = tools.read_txt(measurement_bulk_dir + "\\" + file)

        i = data["intensity"]
        i[i < 0] = 0.0

        I = I + i.to_numpy()
        count = count + 1

    I = I / count

    IAF = np.zeros_like(λ)
    count = 0
    for file in os.listdir(ref_bulk_dir):
        data = tools.read_txt(ref_bulk_dir + "\\" + file)

        iaf = data["intensity"]
        iaf[iaf < 0] = 0.0

        IAF = IAF + iaf.to_numpy()
        count = count + 1

    IAF = IAF / count
    _, IAF = tools.grid_transform(λ, IAF)

    dark = np.zeros_like(λ)
    count = 0
    for file in os.listdir(dark_bulk_dir):
        data = tools.read_txt(dark_bulk_dir + "\\" + file)

        d = data["intensity"]
        d[d < 0] = 0.0

        dark = dark + d.to_numpy()
        count = count + 1

    dark = dark / count
    I = I - dark
    xpw_data = xpw_data - dark
    IAF = IAF - dark

    fig, ax = plt.subplots(1, 2, figsize = (8, 6))
    ax[0].plot(λ, xpw_data / xpw_data.max())
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"$I_X$")

    ω, xpw_data = tools.grid_transform(λ, xpw_data)
    AXω_known = np.sqrt(np.clip(xpw_data, 0.0, None))

    ax[1].plot(ω, xpw_data / xpw_data.max())
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"$I_X$")
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(1, 2, figsize = (8, 6))
    ax[0].plot(λ, I / I.max())
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"I")

    ω, I = tools.grid_transform(λ, I)
    ω0 = tools.centroid_calc_np_1d(I, ω)
    gauss = np.exp(-((ω - ω0)/(2 * 2e14))**10)

    ax[1].plot(ω, I / I.max())
    ax[1].plot(ω, gauss)
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"I")
    plt.tight_layout()
    plt.show()

    I = I * gauss

# Extract AC peak

if simulation:
    I_hat = tools.ft(I_noise)

else:
    I_hat = tools.ft(I)

t = tools.ω2t(ω)
low_AC = 378 * fs
high_AC = 850 * fs
low_DC = -200 * fs
high_DC = 200 * fs

I_abs = np.abs(I_hat / I_hat.max())
fig, ax = plt.subplots(1)
ax.plot(t, I_abs)
ax.vlines([low_AC, high_AC], ymin=0, ymax=1.0, color = "red")
ax.vlines([low_DC, high_DC], ls = "--", ymin=0, ymax=1.0, color = "green")
ax.set_title("|I(t)|")
plt.show()

lower_AC = low_AC
upper_AC = high_AC
mask_AC = (t >= lower_AC) & (t <= upper_AC)
lower_DC = low_DC
upper_DC = high_DC
mask_DC = (t >= lower_DC) & (t <= upper_DC)

I_hat_AC = np.zeros_like(I_hat)
I_hat_DC = np.zeros_like(I_hat)
I_hat_AC[mask_AC] = I_hat[mask_AC]
I_hat_DC[mask_DC] = I_hat[mask_DC]

I_hat_AC = tools.recenter(I_hat_AC, t)

ω0 = tools.centroid_calc_np_1d(np.abs(I_hat_AC), ω)
gauss = np.exp(-((ω - ω0)/(2 * 0.5e14))**10)


fig, ax = plt.subplots(1)
ax.plot(ω, np.abs(I_hat_AC) / np.abs(I_hat_AC).max())
ax.plot(ω, gauss)

plt.show()

I_hat_AC = I_hat_AC * gauss
I_hat_AC = gaussian_filter1d(I_hat_AC, sigma = 3)

ω0 = tools.centroid_calc_np_1d(np.abs(I_hat_DC), ω)
gauss = np.exp(-((ω - ω0)/(4 * 3.5e14))**10)

I_hat_DC = I_hat_DC * gauss

f = tools.ift(I_hat_AC)
S0 = tools.ift(I_hat_DC)

# f = f * gauss

ω0 = tools.centroid_calc_np_1d(np.abs(S0), ω)
gauss = np.ones_like(np.exp(-((ω - ω0)/(6 * 2e14))**10))

fig, ax = plt.subplots(1)
ax.plot(ω, np.abs(S0))

plt.show()

# S0 = S0 * gauss



ids = np.where(f > 1e-6 * f.max())
f = f[ids]
S0 = S0[ids]
ω = ω[ids]
I = I[ids]
IAF = IAF[ids]

if simulation:
    AAω = AAω[ids]

AXω_known = AXω_known[ids]



AFω = tools.get_amplitude(S0, f)
arg_f = np.unwrap(np.angle(f))
abs_f = np.abs(f)

AFω = tools.get_amplitude(S0, f)
φ_in = np.unwrap(np.angle(f)).copy()
φ_xpw = np.zeros_like(φ_in)

max_iter = 10000
tol_amp = 1
tol_phase = 1e-4
alpha = 0.3

for k in range(max_iter):
    # Fundamental in ω and t
    E_in_ω = AFω * np.exp(1j * φ_in)
    E_in_t = tools.ift(E_in_ω)

    # XPW generation (time-domain cubic)
    E_xpw_t_sim = E_in_t * np.abs(E_in_t) ** 2
    E_xpw_ω_sim = tools.ft(E_xpw_t_sim)

    # Use simulated XPW phase + measured XPW amplitude
    φ_xpw = np.unwrap(np.angle(E_xpw_ω_sim))
    E_xpw_ω = AXω_known * np.exp(1j * φ_xpw)

    Δφ = np.unwrap(np.angle(f))
    φ_in_new = Δφ - φ_xpw
    φ_in_new -= φ_in_new[len(φ_in_new) // 2]

    eps = 1e-3 * AXω_known.max()
    denom = np.maximum(AXω_known, eps)

    AFω_target = np.abs(f) / denom
    AFω_new = alpha * AFω_target + (1 - alpha) * AFω

    damp = np.sum((np.abs(AFω) - np.abs(AFω_new)) ** 2) / len(AFω)
    dphi = np.sum((φ_in - φ_in_new) ** 2) / len(φ_in)

    if k % 100 == 0:
        print("=" * 50)
        print(f"Iteration {k}")
        print("=" * 50)
        print(f"φ Error: {dphi}")
        print(f"A Error: {damp}")
        print()

    if dphi < tol_phase and damp < tol_amp:
        print(f"Converged in {k+1} iterations")
        break

    φ_in = φ_in_new
    AFω = AFω_new

# AFω = (np.abs(f).max() / np.abs(AXω_known).max()) * (AFω / AFω.max())
ids = np.where(I > 1e-2 * I.max())
f = f[ids]
S0 = S0[ids]
ω = ω[ids]
I = I[ids]
IAF = IAF[ids]
AXω_known = AXω_known[ids] 
φ_xpw = φ_xpw[ids]
AFω = AFω[ids]
φ_in = φ_in[ids]

E_in_final  = AFω * np.exp(1j * φ_in)
E_xpw_final = AXω_known * np.exp(1j * φ_xpw)

E_xpw_delayed = E_xpw_final * np.exp(1j * ω * τ_delay)

I_sim = np.abs(E_in_final + E_xpw_delayed)**2

fig, ax = plt.subplots(1)
ax.plot(ω, I, label = "Measured Intensity")
ax.plot(ω, I_sim, label = "Computational Intensity")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("I")

plt.legend()
plt.show()

fig, ax = plt.subplots(1)

if simulation:
    ax.plot(ω, np.abs(AAω) ** 2, label = "Real")
    ax.plot(ω, np.abs(IAF - AFω ** 2), "--", label = "Measured")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"I")
    plt.legend()

else:
    ax.plot(ω, np.abs(IAF - AFω ** 2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"I")

plt.show()


print()

