"""
Created on Mon May 21 22:32:40 2025

@author: sab
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Paths
datafolder = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Radii/20250321"
datafolder_counts = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/VoorSaban2/20250321/DataMainSetup"
figfolder = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Figures/20250321/Cu_Si"
os.makedirs(figfolder, exist_ok=True)
materiaal = 'Cu_Si'


# Constants of the setup
picojoules_per_count = 4.3 
pixel_resolution = 26.6 / 100  # Micrometer per pixel
waist = 11.3  # Beam waist in µm
waist_cm = waist * 1e-4
a_fixed = waist**2 / 2  # fixed slope for Liu function
pulse_durations = [0.2, 3.0, 8.0, 14.0]  # Pulse durations of the runs

# Runs 
first_run = 32
last_run = 47
Nruns = last_run - first_run + 1
Ngroups = 4
Nreps = Nruns // Ngroups

#Parameters
exclude_indices = {0: [*range(0,36), 55], 1: [range(0,41)], 2: [range(0,42)], 3: [range(0,42)]}
exclude_last_n_points = 0
min_detections = 3
max_points_per_group = 8

# Load data
radii = np.zeros((Nruns, 64))
counts = np.zeros((Nruns, 64))
run_numbers = range(first_run, last_run + 1)
for i, run in enumerate(run_numbers):
    radii_path = os.path.join(datafolder, f'run_{str(run).zfill(3)}_radii.npy')
    counts_path = os.path.join(datafolder_counts, f'run_{str(run).zfill(3)}_counts.npy')
    radii[i] = np.load(radii_path)[:, 0]
    counts[i] = np.load(counts_path)

# Convert units
radii_sq = (pixel_resolution * radii) ** 2
pjs = counts * picojoules_per_count
radii_sq = radii_sq.reshape((Ngroups, Nreps, 64))
pjs = pjs.reshape((Ngroups, Nreps, 64))

# Liu functions
def liu_func(x, Eth):
    x = np.clip(x, 1e-12, None)
    Eth = max(Eth, 1e-12)
    y = a_fixed * (np.log(x) - np.log(Eth))
    return y * np.heaviside(y, 0)

def liu_func_for_plot(x, Eth):
    y = a_fixed * (np.log(np.clip(x, 1e-12, None)) - np.log(Eth))
    return np.where(x >= Eth, y, 0)

# Averages
avg_rsq = np.zeros((Ngroups, 64))
std_rsq = np.zeros((Ngroups, 64))
for group in range(Ngroups):
    for pixel in range(64):
        values = radii_sq[group, :, pixel]
        valid = values[values > 0]
        if len(valid) >= min_detections:
            avg_rsq[group, pixel] = np.mean(valid)
            std_rsq[group, pixel] = np.std(valid) / np.sqrt(len(valid))

avg_pjs = np.mean(pjs, axis=1)
std_pjs = np.std(pjs, axis=1)

# Plot
colors = plt.cm.cool(np.linspace(0, 1, Ngroups))
plt.figure(figsize=(8, 6))
threshold_data = np.zeros((Ngroups, 5))  # pulse, Eth, Eth_std, Fth, Fth_std

for i in range(Ngroups):
    x = avg_pjs[i][:64 - exclude_last_n_points]
    y = avg_rsq[i][:64 - exclude_last_n_points]
    xerr = std_pjs[i][:64 - exclude_last_n_points]
    yerr = std_rsq[i][:64 - exclude_last_n_points]

    include = np.ones_like(y, dtype=bool)
    include[exclude_indices.get(i, [])] = False
    mask = (y > 0) & include

    xdata, ydata = x[mask], y[mask]
    xerr_data, yerr_data = xerr[mask], yerr[mask]

    if len(xdata) > max_points_per_group:
        xdata, ydata = xdata[:max_points_per_group], ydata[:max_points_per_group]
        xerr_data, yerr_data = xerr_data[:max_points_per_group], yerr_data[:max_points_per_group]

    plt.errorbar(xdata, ydata, xerr=xerr_data, yerr=yerr_data, fmt='o',
                 label=f"{pulse_durations[i]} ps", color=colors[i], capsize=3)

    if len(xdata) < 2:
        print(f"Skipping Group {i}")
        continue

    try:
        popt, pcov = curve_fit(liu_func, xdata, ydata, p0=[np.median(xdata)])
        Eth_pJ, Eth_std_pJ = popt[0], np.sqrt(np.diag(pcov))[0]

        Eth_J = Eth_pJ * 1e-12
        Eth_std_J = Eth_std_pJ * 1e-12
        A_cm2 = 0.5 * np.pi * waist_cm**2

        Fth = Eth_J / A_cm2
        Fth_std = Eth_std_J / A_cm2

        x_range = np.logspace(np.log10(max(1e-1, Eth_pJ / 5)), np.log10(np.max(xdata) * 1.1), 200)
        plt.plot(x_range, liu_func_for_plot(x_range, *popt), '--', color=colors[i])

        print(f"Group {i} ({pulse_durations[i]} ps): Eth = {Eth_pJ:.3f} ± {Eth_std_pJ:.3f} pJ, "
              f"Fth = {Fth:.2e} ± {Fth_std:.2e} J/cm^2")

        threshold_data[i] = [pulse_durations[i], Eth_pJ, Eth_std_pJ, Fth, Fth_std]

    except RuntimeError:
        print(f"Fit failed for group {i}")

# Finalize and save
plt.xscale('log')
plt.ylim([-1, 40])
plt.xlim([9*1e5, 2*1e6])
plt.xlabel("Pulse energy [pJ]", fontsize=14)
plt.ylabel(r"Radius$^2$ [$\mu$m$^2$]", fontsize=14)
plt.legend(title="Pulse duration", loc =2)
plt.tight_layout()

pdf_path = os.path.join(figfolder, f"liu_data_{materiaal}_{first_run}_{last_run}.pdf")
np.save(os.path.join(figfolder, f"liu_thresholds_{first_run}_w-fixed_{last_run}.npy"), threshold_data)
plt.savefig(pdf_path)
plt.show()
print(f"PDF saved to: {pdf_path}")
