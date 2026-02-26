"""
Created on Mon May 21 22:32:40 2025

@author: sab
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Paths
datafolder = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Method 1/RadiiAndCoords/20250321"
datafolder_counts = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/VoorSaban2/20250321/DataMainSetup"
figfolder = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Method 1/Results/PDFs"
os.makedirs(figfolder, exist_ok=True)
materiaal = 'Cu_Si'

# Constants
picojoules_per_count = 4.3 
pixel_resolution = 26.6 / 100  # Micrometer per pixel
waist = 11.3  # Beam waist in µm
waist_cm = waist * 1e-4
a_fixed = waist**2 / 2  # Liu slope constant
pulse_durations = [0.2, 3.0, 8.0, 14.0]

# Runs
first_run = 32
last_run = 47
Nruns = last_run - first_run + 1
Ngroups = 4
Nreps = Nruns // Ngroups

# Settings
exclude_indices = {0: [], 1: [], 2: [], 3: []}
exclude_last_n_points = 0
min_detections = 3
max_points_per_group = 8

# Load data
radii = np.zeros((Nruns, 64))
counts = np.zeros((Nruns, 64))
run_numbers = range(first_run, last_run + 1)
for i, run in enumerate(run_numbers):
    radii_path = os.path.join(datafolder, f'run_{str(run).zfill(3)}_radii_coords.npy')
    counts_path = os.path.join(datafolder_counts, f'run_{str(run).zfill(3)}_counts.npy')
    radii_data = np.load(radii_path)
    radii[i] = radii_data[:, 0]
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
    x = np.clip(x, 1e-12, None)
    y = a_fixed * (np.log(x) - np.log(Eth))
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
        else:
            avg_rsq[group, pixel] = 0
            std_rsq[group, pixel] = 0

avg_pjs = np.mean(pjs, axis=1)
std_pjs = np.std(pjs, axis=1)

# Plotting
colors = plt.cm.cool(np.linspace(0, 1, Ngroups))
plt.figure(figsize=(8, 6))

threshold_data = np.zeros((Ngroups, 5))  # [pulse_duration, Eth_pJ, Eth_std_pJ, Fth, Fth_std]

for i in range(Ngroups):
    end = 64 - exclude_last_n_points
    x_full = avg_pjs[i][:end]
    y_full = avg_rsq[i][:end]
    xerr_full = std_pjs[i][:end]
    yerr_full = std_rsq[i][:end]

    exclude_idx = exclude_indices.get(i, [])
    include = np.ones_like(y_full, dtype=bool)
    include[exclude_idx] = False
    nonzero = (y_full > 0) & include

    x_plot = x_full[nonzero][:max_points_per_group]
    y_plot = y_full[nonzero][:max_points_per_group]
    xerr_plot = xerr_full[nonzero][:max_points_per_group]
    yerr_plot = yerr_full[nonzero][:max_points_per_group]

    plt.errorbar(x_plot, y_plot, xerr=xerr_plot, yerr=yerr_plot,
                 fmt='o', label=f'{pulse_durations[i]} ps',
                 color=colors[i], capsize=3, alpha=0.8)

    if len(x_plot) < 2:
        print(f"Skipping Group {i} due to insufficient data.")
        continue

    try:
        popt, pcov = curve_fit(liu_func, x_plot, y_plot, p0=[np.median(x_plot)])
        perr = np.sqrt(np.diag(pcov))

        Eth_pJ = popt[0]
        Eth_std_pJ = perr[0]

        Eth_J = Eth_pJ * 1e-12
        Eth_std_J = Eth_std_pJ * 1e-12

        Fth = Eth_J / (0.5 * np.pi * waist_cm**2)
        Fth_std = Eth_std_J / (0.5 * np.pi * waist_cm**2)

        x_range = np.logspace(np.log10(max(1e-1, Eth_pJ / 5)), np.log10(np.max(x_plot) * 1.1), 200)
        fit_y = liu_func_for_plot(x_range, *popt)
        plt.plot(x_range, fit_y, linestyle='--', color=colors[i])

        print(f"Group {i} ({pulse_durations[i]} ps): Eth = {Eth_pJ:.3f} ± {Eth_std_pJ:.3f} pJ, "
              f"Fth = {Fth:.2e} ± {Fth_std:.2e} J/cm^2")

        threshold_data[i] = [pulse_durations[i], Eth_pJ, Eth_std_pJ, Fth, Fth_std]

    except RuntimeError:
        print(f"Fit failed for Group {i}.")

# Final plot settings
plt.xscale('log')
plt.xlabel("Pulse energy [pJ]", fontsize=14)
plt.ylabel(r"Radius$^2$ [$\mu$m$^2$]", fontsize=14)
# plt.ylim([-1, 40])
# plt.xlim([0.8*1e5, 0.8*1e6])
plt.grid(False)
plt.legend(title="Pulse duration", loc = 2)
plt.tight_layout()

# Save figure
plot_path = os.path.join(figfolder, f"liu_data_{materiaal}_{first_run}_{last_run}_Marnix1_bruh.pdf")
plt.savefig(plot_path)
plt.show()

# Save threshold data
threshold_file = os.path.join(figfolder, f"liu_thresholds_{first_run}_w-fixed_{last_run}_marnix_1.npy")
np.save(threshold_file, threshold_data)
