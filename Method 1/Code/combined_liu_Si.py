import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# ============================
# Setup & Constants
# ============================

materiaal = 'Si'
first_run = 24
last_run = 39
Nruns = last_run - first_run + 1
Ngroups = 4
Nreps = Nruns // Ngroups
pulse_durations = [0.2, 3.0, 8.0, 14.0]
selected_groups = [0, 3]  # Only 0.2 ps and 14 ps

picojoules_per_count = 4.3 
pixel_resolution = 26.6 / 100  # µm/pixel
waist = 11.3  # µm
waist_cm = waist * 1e-4
a_fixed = waist**2 / 2
min_detections = 3
max_points_per_group = 8

def liu_func(x, Eth):
    x = np.clip(x, 1e-12, None)
    y = a_fixed * (np.log(x) - np.log(Eth))
    return y * np.heaviside(y, 0)

def liu_func_for_plot(x, Eth):
    y = a_fixed * (np.log(np.clip(x, 1e-12, None)) - np.log(Eth))
    return np.where(x >= Eth, y, 0)

# ============================
# Data Loading Function
# ============================

def load_and_process(datafolder, counts_folder, suffix, exclude_dict):
    radii = np.zeros((Nruns, 64))
    counts = np.zeros((Nruns, 64))
    for i, run in enumerate(range(first_run, last_run + 1)):
        f = f'run_{str(run).zfill(3)}_{suffix}.npy'
        radii[i] = np.load(os.path.join(datafolder, f))[:, 0]
        counts[i] = np.load(os.path.join(counts_folder, f'run_{str(run).zfill(3)}_counts.npy'))

    radii_sq = (pixel_resolution * radii)**2
    pjs = counts * picojoules_per_count
    radii_sq = radii_sq.reshape((Ngroups, Nreps, 64))
    pjs = pjs.reshape((Ngroups, Nreps, 64))

    avg_rsq, std_rsq = np.zeros((Ngroups, 64)), np.zeros((Ngroups, 64))
    for group in range(Ngroups):
        excluded = exclude_dict.get(group, [])
        for pixel in range(64):
            if pixel in excluded:
                continue
            vals = radii_sq[group, :, pixel]
            valid = vals[vals > 0]
            if len(valid) >= min_detections:
                avg_rsq[group, pixel] = np.mean(valid)
                std_rsq[group, pixel] = np.std(valid) / np.sqrt(len(valid))

    avg_pjs = np.mean(pjs, axis=1)
    std_pjs = np.std(pjs, axis=1)
    return avg_rsq, std_rsq, avg_pjs, std_pjs

# ============================
# Load Data
# ============================

auto_data = load_and_process(
    "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Radii/20250320",
    "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/VoorSaban2/20250320/DataMainSetup",
    "radii",
    {0: [], 1: [], 2: [], 3: [25, 28]}
)

thresh_data = load_and_process(
    "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Method 1/RadiiAndCoords/20250320",
    "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/VoorSaban2/20250320/DataMainSetup",
    "radii_coords",
    {0: [], 1: [], 2: [], 3: list(range(29, 64))}
)

avg_rsq_auto, std_rsq_auto, avg_pjs_auto, std_pjs_auto = auto_data
avg_rsq_thresh, std_rsq_thresh, avg_pjs_thresh, std_pjs_thresh = thresh_data

# ============================
# Plotting
# ============================

blues = plt.cm.Blues(np.linspace(0.75, 0.95, 2))
oranges = plt.cm.Oranges(np.linspace(0.75, 0.95, 2))

plt.figure(figsize=(9, 6))

for plot_idx, i in enumerate(selected_groups):
    pulse_label = f"{pulse_durations[i]} ps"
    color_auto = blues[plot_idx]
    color_thresh = oranges[plot_idx]

    # Automated
    x_auto, y_auto = avg_pjs_auto[i], avg_rsq_auto[i]
    xerr_auto, yerr_auto = std_pjs_auto[i], std_rsq_auto[i]
    mask_auto = y_auto > 0
    x_auto, y_auto = x_auto[mask_auto][:max_points_per_group], y_auto[mask_auto][:max_points_per_group]
    xerr_auto, yerr_auto = xerr_auto[mask_auto][:max_points_per_group], yerr_auto[mask_auto][:max_points_per_group]

    plt.errorbar(x_auto, y_auto, xerr=xerr_auto, yerr=yerr_auto,
                 fmt='^', color=color_auto, markeredgecolor='black',
                 label=f"{pulse_label} (Auto)", capsize=3, markersize=8)

    try:
        popt, _ = curve_fit(liu_func, x_auto, y_auto, p0=[np.median(x_auto)])
        x_fit = np.logspace(np.log10(popt[0]/5), np.log10(np.max(x_auto)*1.1), 200)
        plt.plot(x_fit, liu_func_for_plot(x_fit, *popt), '-', color=color_auto)
    except RuntimeError:
        print(f"Fit failed for Auto group {i}")

    # Threshold
    x_thr, y_thr = avg_pjs_thresh[i], avg_rsq_thresh[i]
    xerr_thr, yerr_thr = std_pjs_thresh[i], std_rsq_thresh[i]
    mask_thr = y_thr > 0
    x_thr, y_thr = x_thr[mask_thr][:max_points_per_group], y_thr[mask_thr][:max_points_per_group]
    xerr_thr, yerr_thr = xerr_thr[mask_thr][:max_points_per_group], yerr_thr[mask_thr][:max_points_per_group]

    plt.errorbar(x_thr, y_thr, xerr=xerr_thr, yerr=yerr_thr,
                 fmt='v', color=color_thresh, markeredgecolor='black',
                 label=f"{pulse_label} (Thresh)", capsize=3, markersize=8)

    try:
        popt, _ = curve_fit(liu_func, x_thr, y_thr, p0=[np.median(x_thr)])
        x_fit = np.logspace(np.log10(popt[0]/5), np.log10(np.max(x_thr)*1.1), 200)
        plt.plot(x_fit, liu_func_for_plot(x_fit, *popt), '--', color=color_thresh)
    except RuntimeError:
        print(f"Fit failed for Thresh group {i}")

# ============================
# Final Touches
# ============================

plt.xscale('log')
plt.xlim([0.9e5, 2.8e6])
plt.ylim([-1, 70])
plt.xlabel("Pulse energy [pJ]", fontsize=14)
plt.ylabel(r"Radius$^2$ [$\mu$m$^2$]", fontsize=14)
plt.legend(title="Pulse duration + Method", fontsize=10)
plt.tight_layout()

# ============================
# Save
# ============================

save_folder = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Figures/20250320/combined plots"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, f"liu_combined_2lines_colormaps_{materiaal}_{first_run}_{last_run}.svg")
plt.savefig(save_path)
plt.show()
print(f"✅ Combined 2-line colormap plot saved to: {save_path}")
