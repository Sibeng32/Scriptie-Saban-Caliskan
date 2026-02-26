import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# ============================
# Setup
# ============================

materiaal = 'SOI'
first_run = 40
last_run = 55
Nruns = last_run - first_run + 1
Ngroups = 4
Nreps = Nruns // Ngroups
pulse_durations = [0.2, 3.0, 8.0, 14.0]

selected_groups = [0, 3]  # Only 0.2 ps and 14 ps

picojoules_per_count = 4.3 
pixel_resolution = 26.6 / 100
waist = 11.3
waist_cm = waist * 1e-4
a_fixed = waist**2 / 2
min_detections = 3
max_points_per_group = 8

# Liu functions
def liu_func(x, Eth):
    x = np.clip(x, 1e-12, None)
    Eth = max(Eth, 1e-12)
    y = a_fixed * (np.log(x) - np.log(Eth))
    return y * np.heaviside(y, 0)

def liu_func_for_plot(x, Eth):
    y = a_fixed * (np.log(np.clip(x, 1e-12, None)) - np.log(Eth))
    return np.where(x >= Eth, y, 0)

# ============================
# Data Loaders
# ============================

def load_data(method, radii_suffix):
    basefolder = {
        "auto": "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Radii/20250320",
        "thresh": "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Method 1/RadiiAndCoords/20250320"
    }[method]
    
    counts_folder = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/VoorSaban2/20250320/DataMainSetup"

    radii = np.zeros((Nruns, 64))
    counts = np.zeros((Nruns, 64))
    for i, run in enumerate(range(first_run, last_run + 1)):
        rpath = os.path.join(basefolder, f'run_{str(run).zfill(3)}_{radii_suffix}.npy')
        radii[i] = np.load(rpath)[:, 0]
        counts[i] = np.load(os.path.join(counts_folder, f'run_{str(run).zfill(3)}_counts.npy'))

    rsq = (pixel_resolution * radii) ** 2
    pjs = counts * picojoules_per_count
    return rsq.reshape((Ngroups, Nreps, 64)), pjs.reshape((Ngroups, Nreps, 64))

# ============================
# Load Data
# ============================

radii_sq_auto, pjs_auto = load_data("auto", "radii")
radii_sq_thresh, pjs_thresh = load_data("thresh", "radii_coords")

def compute_avg_std(radii_sq, pjs):
    avg_rsq = np.zeros((Ngroups, 64))
    std_rsq = np.zeros((Ngroups, 64))
    for group in range(Ngroups):
        for pixel in range(64):
            vals = radii_sq[group, :, pixel]
            valid = vals[vals > 0]
            if len(valid) >= min_detections:
                avg_rsq[group, pixel] = np.mean(valid)
                std_rsq[group, pixel] = np.std(valid) / np.sqrt(len(valid))
    return avg_rsq, std_rsq, np.mean(pjs, axis=1), np.std(pjs, axis=1)

avg_rsq_auto, std_rsq_auto, avg_pjs_auto, std_pjs_auto = compute_avg_std(radii_sq_auto, pjs_auto)
avg_rsq_thresh, std_rsq_thresh, avg_pjs_thresh, std_pjs_thresh = compute_avg_std(radii_sq_thresh, pjs_thresh)

# ============================
# Plot
# ============================

blues = plt.cm.Blues(np.linspace(0.75, 0.95, 2))
oranges = plt.cm.Oranges(np.linspace(0.75, 0.95, 2))


plt.figure(figsize=(9, 6))

for plot_idx, i in enumerate(selected_groups):
    pulse_label = f"{pulse_durations[i]} ps"
    color_auto = blues[plot_idx]
    color_thresh = oranges[plot_idx]

    # --- Automated ---
    x_auto, y_auto = avg_pjs_auto[i], avg_rsq_auto[i]
    xerr_auto, yerr_auto = std_pjs_auto[i], std_rsq_auto[i]
    mask_auto = (y_auto > 0)
    x_auto, y_auto = x_auto[mask_auto][:max_points_per_group], y_auto[mask_auto][:max_points_per_group]
    xerr_auto, yerr_auto = xerr_auto[mask_auto][:max_points_per_group], yerr_auto[mask_auto][:max_points_per_group]

    plt.errorbar(x_auto, y_auto, xerr=xerr_auto, yerr=yerr_auto,
                 fmt='^', color=color_auto, markersize=8, markeredgecolor='black',
                 capsize=3, label=f"{pulse_label} (Auto)")
    try:
        popt, _ = curve_fit(liu_func, x_auto, y_auto, p0=[np.median(x_auto)])
        x_fit = np.logspace(np.log10(popt[0]/5), np.log10(np.max(x_auto)*1.1), 200)
        plt.plot(x_fit, liu_func_for_plot(x_fit, *popt), '-', color=color_auto)
    except RuntimeError:
        print(f"Auto fit failed for group {i}")

    # --- Threshold ---
    x_thr, y_thr = avg_pjs_thresh[i], avg_rsq_thresh[i]
    xerr_thr, yerr_thr = std_pjs_thresh[i], std_rsq_thresh[i]
    mask_thr = (y_thr > 0)
    x_thr, y_thr = x_thr[mask_thr][:max_points_per_group], y_thr[mask_thr][:max_points_per_group]
    xerr_thr, yerr_thr = xerr_thr[mask_thr][:max_points_per_group], yerr_thr[mask_thr][:max_points_per_group]

    plt.errorbar(x_thr, y_thr, xerr=xerr_thr, yerr=yerr_thr,
                 fmt='v', color=color_thresh, markersize=8,
                 capsize=3, markeredgecolor='black', label=f"{pulse_label} (Thresh)")
    try:
        popt, _ = curve_fit(liu_func, x_thr, y_thr, p0=[np.median(x_thr)])
        x_fit = np.logspace(np.log10(popt[0]/5), np.log10(np.max(x_thr)*1.1), 200)
        plt.plot(x_fit, liu_func_for_plot(x_fit, *popt), '--', color=color_thresh)
    except RuntimeError:
        print(f"Threshold fit failed for group {i}")

# ============================
# Finalize
# ============================

plt.xscale('log')
plt.xlim([0.9e5, 1.1e6])
plt.ylim([-1, 70])
plt.xlabel("Pulse energy [pJ]", fontsize=14)
plt.ylabel(r"Radius$^2$ [$\mu$m$^2$]", fontsize=14)
plt.legend(title="Pulse duration + Method", fontsize=10)
plt.tight_layout()

# ============================
# Save
# ============================

figfolder = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Figures/20250320/combined plots"
os.makedirs(figfolder, exist_ok=True)
filename = f"liu_combined_2lines_colormaps_{materiaal}_{first_run}_{last_run}.svg"
plt.savefig(os.path.join(figfolder, filename))
plt.show()

print(f"Saved combined plot with 2 colormaps to: {os.path.join(figfolder, filename)}")
