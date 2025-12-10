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
    Δω = 2 * np.pi * c * Δλ / λ0 ** 2

    τ_delay = 500 * fs
    
    AF = np.exp(-(λ - λ0)**2 / Δλ**2)
    AA = 0.50 * AF
    ω, AFω = tools.grid_transform(λ, AF)
    _, AAω = tools.grid_transform(λ, AA)
    dω = ω[1] - ω[0]
    dt = 2 * np.pi / (N * dω)
    t = (np.arange(N) - N // 2) * dt
    dω = ω[1] - ω[0]
    ω_fft = (np.arange(N) - N // 2) * dω   # FFT-conjugate grid
    #φ0 = np.zeros_like(ω)

    ω0 = tools.centroid_calc_np_1d(AFω, ω)
    φ0 = np.zeros_like((10 * fs) ** 3 * (ω - ω0) ** 3)

    EFω_true = AFω * np.exp(1j * φ0)
    EF  = tools.ft_norm(EFω_true, dx=dω)
    EX_true  = EF * np.abs(EF)**2
    EXω_true = tools.ift_norm(EX_true, dx=dω)
    EXω_true = 0.3 * EXω_true / (np.abs(EXω_true).max() * AFω.max())
    AXω_known = np.abs(EXω_true)

    I = np.abs(AFω) ** 2 + np.abs(AAω) ** 2 * 0 + np.abs(AXω_known) ** 2 + 2 * AFω * AXω_known * np.cos(ω * τ_delay + (φ0 - np.unwrap(np.angle(EXω_true))))
    I_noise = I + 1e-6 * (np.random.rand(I.size) - 0.5)

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
    dω = ω[1] - ω[0]
    AXω_known = np.sqrt(np.clip(xpw_data, 0.0, None))

    ax[1].plot(ω, xpw_data / xpw_data.max())
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"$I_X$")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize = (8, 6))
    ax[0].plot(λ, IAF / IAF.max())
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"IAF")

    _, IAF = tools.grid_transform(λ, IAF)

    ax[1].plot(ω, IAF / IAF.max())
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"IAF")
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
    # ax[1].plot(ω, gauss)
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"I")
    plt.tight_layout()
    plt.show()

    I = I * gauss

# Extract AC peak

if simulation:
    I_hat = tools.ft_norm(I_noise, dx=dω)

else:
    I_hat = tools.ft_norm(I, dx=dω)

t = tools.ω2t(ω)
(low_DC_idx, high_DC_idx), (low_AC_idx, high_AC_idx) = tools.ac_dc_identify(I_hat, t, dc_radius=50)
#low_AC = 378 * fs
#high_AC = 850 * fs
#low_DC = -200 * fs
#high_DC = 200 * fs

I_abs = np.abs(I_hat / I_hat.max())
fig, ax = plt.subplots(1)
ax.plot(t, I_abs)
ax.vlines([t[low_AC_idx], t[high_AC_idx]], ymin=0, ymax=1.0, color = "red")
ax.vlines([t[low_DC_idx], t[high_DC_idx]], ls = "--", ymin=0, ymax=1.0, color = "green")
ax.set_title("|I(t)|")
plt.show()

#lower_AC = low_AC
#upper_AC = high_AC
mask_AC = (t >= t[low_AC_idx]) & (t <= t[high_AC_idx])
#lower_DC = low_DC
#upper_DC = high_DC
mask_DC = (t >= t[low_DC_idx]) & (t <= t[high_DC_idx])

I_hat_AC = np.zeros_like(I_hat)
I_hat_DC = np.zeros_like(I_hat)
I_hat_AC[mask_AC] = I_hat[mask_AC]
I_hat_DC[mask_DC] = I_hat[mask_DC]

I_hat_AC = tools.recenter(I_hat_AC, t)


ids = np.where(np.abs(I) > 1e-2 * np.abs(I).max())
test_omega = ω[ids]
test_AC = tools.ift_norm(I_hat_AC, dx=dω)
angle = np.unwrap(np.angle(test_AC))

ω0 = tools.centroid_calc_np_1d(np.abs(I_hat_AC), ω)
gauss = np.exp(-((ω - ω0)/(2 * 0.5e14))**10)


# fig, ax = plt.subplots(1)
# ax.plot(ω, np.abs(I_hat_AC) / np.abs(I_hat_AC).max())
# ax.plot(ω, gauss)

# plt.show()

I_hat_AC = I_hat_AC * gauss

# fig, ax = plt.subplots(1)
# ax.plot(ω, np.abs(I_hat_AC) / np.abs(I_hat_AC).max())
# plt.show()


# I_hat_AC = gaussian_filter1d(I_hat_AC, sigma = 1)

fig, ax = plt.subplots(1)
ax.plot(ω, np.abs(I_hat_AC) / np.abs(I_hat_AC).max())
plt.show()

# ω0 = tools.centroid_calc_np_1d(np.abs(I_hat_DC), ω)
gauss = np.exp(-((ω - ω0)/(4 * 3.5e14))**10)

I_hat_DC = I_hat_DC * gauss

f = tools.ift_norm(I_hat_AC, dx=dω)
S0 = tools.ift_norm(I_hat_DC, dx=dω)

# f = f * gauss

# ω0 = tools.centroid_calc_np_1d(np.abs(S0), ω)
# gauss = np.ones_like(np.exp(-((ω - ω0)/(6 * 2e14))**10))

fig, ax = plt.subplots(1)
ax.plot(ω, np.abs(S0))

plt.show()

# S0 = S0 * gauss



# ids = np.where(f > 1e-6 * f.max())
# f = f[ids]
# S0 = S0[ids]
# ω = ω[ids]
# I = I[ids]
# IAF = IAF[ids]

# if simulation:
#     AAω = AAω[ids]

# AXω_known = AXω_known[ids]



AFω = tools.get_amplitude(S0, f)
Δφ = np.unwrap(np.angle(f))
abs_f = np.abs(f)

AFω = tools.get_amplitude(S0, f)
φ_in = np.unwrap(np.angle(f)).copy()
φ_xpw = np.zeros_like(φ_in)

# phase = np.exp(1j * np.unwrap(np.angle(f)))
# phase_F = phase.copy()
# phase_X = np.exp(1j * np.zeros_like(phase_F))

max_iter = 10000
tol_amp = 1e-6
tol_phase = 1e-6
alpha = 0.3
ε = 1e-10

for k in range(max_iter):

    # Fundamental in ω and t
    E_in_ω = AFω * np.exp(1j * φ_in)
    # E_in_ω = AFω * phase_F
    E_in_t = tools.ift_norm(E_in_ω, dx=dω)

    # XPW generation (time-domain cubic)
    E_xpw_t_sim = E_in_t * np.abs(E_in_t) ** 2
    E_xpw_ω_sim = tools.ft_norm(E_xpw_t_sim, dx=dω)
    φ_xpw = np.angle(E_xpw_ω_sim)

    # Use simulated XPW phase + measured XPW amplitude
    denom = AXω_known + ε

    # phase_X = E_xpw_ω_sim / denom
    # phase_X = phase_X / np.abs(phase_X)
    # E_xpw_ω = AXω_known * phase_X

    φ_in_new = Δφ - φ_xpw
    φ_in_new -= φ_in_new[len(φ_in_new) // 2]

    # phase_F_new = phase * np.conj(phase_X)
    # phase_F_new = phase_F_new / np.abs(phase_F_new)
    
    AFω_target = np.abs(f) / denom
    AFω_new = alpha * AFω_target + (1 - alpha) * AFω

    damp = np.sum((np.abs(AFω) - np.abs(AFω_new)) ** 2) / len(AFω)
    # dphase = np.sum(np.unwrap(np.angle(phase_F * np.conj(phase_F_new))) ** 2) / len(phase_F)
    dphase = np.sum((φ_in_new - φ_in) ** 2) / len(φ_in)

    if k % 500 == 0:
        print("=" * 50)
        print(f"Iteration {k}")
        print("=" * 50)
        print(f"φ Error: {dphase}")
        print(f"A Error: {damp}")
        print()

    if dphase < tol_phase and damp < tol_amp:
        print(f"Converged in {k+1} iterations")
        break

    φ_in = φ_in_new
    
    # phase_F = phase_F_new
    AFω = AFω_new

# AFω = (np.abs(f).max() / np.abs(AXω_known).max()) * (AFω / AFω.max())

ids = np.where(np.abs(I) > 1e-2 * np.abs(I).max())
f = f[ids]
S0 = S0[ids]
ω = ω[ids]
I = I[ids]
IAF = IAF[ids]
AXω_known = AXω_known[ids] 
φ_xpw = np.unwrap(φ_xpw[ids])
# phase_X = phase_X[ids]
AFω = AFω[ids]
# phase_F = phase_F[ids]
φ_in = np.unwrap(φ_in[ids])

E_F_final  = AFω * np.exp(1j * φ_in)
E_xpw_final = AXω_known * np.exp(1j * φ_xpw)

# E_F_final  = AFω * phase_F
# E_xpw_final = AXω_known * phase_X

E_xpw_delayed = E_xpw_final * np.exp(1j * ω * τ_delay)

I_sim = np.abs(E_F_final + E_xpw_delayed)**2

fig, ax = plt.subplots(1, 2)
ax[0].plot(ω, I, label = "Measured Intensity")
ax[0].plot(ω, I_sim, label = "Computational Intensity")
ax[0].set_xlabel(r"$\omega$")
ax[0].set_ylabel("I")
ax[0].legend()

# φ_F = np.unwrap(np.angle(phase_F))

if simulation:
    ax[1].plot(ω, φ0[ids])
    ax[1].plot(ω, φ_in, "--")

else:
    ax[1].plot(ω, φ_in)

ax[1].set_xlabel(r"$\omega$")
ax[1].set_ylabel(r"$\phi$")


plt.show()

IAA_comp = np.abs(IAF - AFω ** 2)
λ_sample, IAA_comp_λ = tools.grid_transform(ω, IAA_comp)

if simulation:
    fig, ax = plt.subplots(1)
    ax.plot(ω, np.abs(AAω[ids]) ** 2 / (np.abs(AAω) ** 2).max(), label = "Real")
    ax.plot(ω, IAA_comp / IAA_comp.max(), "--", label = "Measured")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"I")
    plt.legend()

else:
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(ω, IAA_comp)
    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel(r"I")
    ax[1].plot(λ_sample, IAA_comp_λ)
    ax[1].set_xlabel(r"$\lambda$")

plt.show()


print()

