"""
Created on Mon May 21 22:32:40 2025

@author: sab
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Paths
datafolder = "//Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Radii/20250321"
figfolder = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Figures/20250321/032_047"
os.makedirs(figfolder, exist_ok=True)
materiaal = 'Cu_Si'

# Constants of the setup
picojoules_per_count = 4.3 
pixel_resolution = 26.6 / 100  # Micrometer per pixel
waist = 11.3  # Beam waist in µm
a_fixed = waist**2 / 2  # fixed slope for Liu function
pulse_durations = [0.2, 3.0, 8.0, 14.0]  # Pulse durations of the runs

# Runs 
first_run = 32
last_run = 47
Nruns = last_run - first_run + 1
Ngroups = 4  # Groups of runs with same pulse duration
Nreps = Nruns // Ngroups 

# Settings
exclude_indices = {0: [range(1,36)], 1: [range(1,40)], 2: [range(1,40)], 3: [range(1,44)]}
exclude_last_n_points = 0
min_detections = 1  # min number of reps where a radius must be found
max_points_per_group = 8  # Limit to first N non-zero points for fit and plot

# Get data 
radii = np.zeros((Nruns, 64))
counts = np.zeros((Nruns, 64))
run_numbers = range(first_run, last_run + 1)
for i in range(len(run_numbers)):
    run = run_numbers[i]
    radii[i] = np.load(os.path.join(datafolder, f'run_{str(run).zfill(3)}_radii.npy'))
    counts[i] = np.load(os.path.join(datafolder, f'run_{str(run).zfill(3)}_counts.npy'))

# Convert units
radii_sq = (pixel_resolution * radii) ** 2
pjs = counts * picojoules_per_count

# Reshape data into 3D arrays
radii_sq = radii_sq.reshape((Ngroups, Nreps, 64))
pjs = pjs.reshape((Ngroups, Nreps, 64))

# Liu Function
def liu_func(x, Eth):
    x = np.clip(x, 1e-12, None)
    Eth = max(Eth, 1e-12)
    y = a_fixed * (np.log(x) - np.log(Eth))
    return y * np.heaviside(y, 0)

def liu_func_for_plot(x, Eth):
    x = np.clip(x, 1e-12, None)
    y = a_fixed * (np.log(x) - np.log(Eth))
    return np.where(x >= Eth, y, 0)

# Compute averages
avg_rsq = np.zeros((Ngroups, 64))
std_rsq = np.zeros((Ngroups, 64))

for group in range(Ngroups):
    for pixel in range(64):
        values = radii_sq[group, :, pixel]
        valid_values = values[values > 0]
        if len(valid_values) >= min_detections:
            avg_rsq[group, pixel] = np.mean(valid_values)
            std_rsq[group, pixel] = np.std(valid_values) / np.sqrt(len(valid_values))
        else:
            avg_rsq[group, pixel] = 0
            std_rsq[group, pixel] = 0

avg_pjs = np.mean(pjs, axis=1)
std_pjs = np.std(pjs, axis=1)

# Plot setup
colors = plt.cm.viridis(np.linspace(0, 1, Ngroups))
plt.figure(figsize=(8, 6))

for i in range(Ngroups):
    end_index = 64 - exclude_last_n_points
    x_full = avg_pjs[i][:end_index]
    y_full = avg_rsq[i][:end_index]
    xerr_full = std_pjs[i][:end_index]
    yerr_full = std_rsq[i][:end_index]

    exclude_idx = exclude_indices.get(i, [])
    include_mask = np.ones_like(y_full, dtype=bool)
    include_mask[exclude_idx] = False
    nonzero_mask = (y_full > 0) & include_mask

    x_plot = x_full[nonzero_mask]
    y_plot = y_full[nonzero_mask]
    xerr_plot = xerr_full[nonzero_mask]
    yerr_plot = yerr_full[nonzero_mask]

    # Limit to first N non-zero points
    if x_plot.size > max_points_per_group:
        x_plot = x_plot[:max_points_per_group]
        y_plot = y_plot[:max_points_per_group]
        xerr_plot = xerr_plot[:max_points_per_group]
        yerr_plot = yerr_plot[:max_points_per_group]

    plt.errorbar(x_plot, y_plot,
                 xerr=xerr_plot, yerr=yerr_plot,
                 fmt='o', label=f'{pulse_durations[i]} ps',
                 color=colors[i], capsize=3, alpha=0.8)

# Fit Liu function
threshold_data = np.zeros((Ngroups, 3))  # [pulse_duration, Eth, Eth_std]

for i in range(Ngroups):
    end_index = 64 - exclude_last_n_points
    x_full = avg_pjs[i][:end_index]
    y_full = avg_rsq[i][:end_index]

    exclude_idx = exclude_indices.get(i, [])
    include_mask = np.ones_like(y_full, dtype=bool)
    include_mask[exclude_idx] = False
    nonzero_mask = (y_full > 0) & include_mask

    xdata = x_full[nonzero_mask]
    ydata = y_full[nonzero_mask]

    # Limit to first N non-zero points
    if xdata.size > max_points_per_group:
        xdata = xdata[:max_points_per_group]
        ydata = ydata[:max_points_per_group]

    if xdata.size < 2:
        print(f"Skipping Group {i} due to insufficient nonzero data for fitting.")
        continue

    try:
        popt, pcov = curve_fit(liu_func, xdata, ydata, p0=[np.median(xdata)])
        perr = np.sqrt(np.diag(pcov))  # standard deviation

        Eth_pJ = popt[0]
        Eth_std_pJ = perr[0]

        x_min = max(1e-1, Eth_pJ / 5)
        x_max = np.max(xdata) * 1.1
        x_range_local = np.logspace(np.log10(x_min), np.log10(x_max), 200)

        fit_y = liu_func_for_plot(x_range_local, *popt)
        plt.plot(x_range_local, fit_y, linestyle='--', color=colors[i])

        Eth_J = Eth_pJ * 1e-12
        waist_cm = waist * 1e-4
        Fth = Eth_J / (0.5 * np.pi * waist_cm**2)

        print(f"Group {i} ({pulse_durations[i]} ps): Eth = {Eth_pJ:.3f} ± {Eth_std_pJ:.3f} pJ, Fth = {Fth:.2e} J/cm^2")

        threshold_data[i, 0] = pulse_durations[i]
        threshold_data[i, 1] = Eth_pJ
        threshold_data[i, 2] = Eth_std_pJ

    except RuntimeError:
        print(f"Curve fitting failed for Group {i}. Skipping fit.")

# Finalize plot
plt.xscale('log')
plt.xlabel("Pulse energy [pJ]")
plt.xlim([0.9*1e6, 2.1*1e6])
plt.ylabel(r"Radius$^2$ [$\mu$m$^2$]")
plt.title(f"Liu Plots for {materiaal}")
plt.grid(False)
plt.legend(title="Pulse duration", loc=2)
plt.tight_layout()

# Save
plot_path = os.path.join(figfolder, f"liu_data_{materiaal}_{first_run}_{last_run}.pdf")
plt.savefig(plot_path)
plt.show()

threshold_file = os.path.join(figfolder, f"liu_thresholds_{first_run}_w-fixed_{last_run}.npy")
np.save(threshold_file, threshold_data)
