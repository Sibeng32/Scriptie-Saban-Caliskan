#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 02:09:51 2025

@author: sab
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
materials = [
    ("Si", "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Figures/20250320/024_039/liu_thresholds_24_w-fixed_39.npy"),
    ("SOI", "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Figures/20250320/040_055/liu_thresholds_40_w-fixed_55.npy"),
    ("a-C", "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Figures/20250321/016_031/liu_thresholds_16_w-fixed_31.npy"),
    ("Cu/SiO2", "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Figures/20250321/032_047/liu_thresholds_32_w-fixed_47.npy")
]

output_path = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Figures/Summary_plot_fluence_thresholds.pdf"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- Plotting ---
plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(materials)))

for i, (material_name, file_path) in enumerate(materials):
    if os.path.exists(file_path):
        data = np.load(file_path)

        pulse_durations = data[:, 0]
        thresholds = data[:, 1]
        threshold_stds = data[:, 2] if data.shape[1] > 2 else np.zeros_like(thresholds)

        # Debug print
        print(f"{material_name} - Threshold stds: {threshold_stds}")

        # Optional: set a floor for tiny stds just to make bars visible
        visible_stds = np.maximum(threshold_stds, 1e-2)

        # Plot error bars separately for visibility
        plt.errorbar(
            pulse_durations, thresholds, yerr=visible_stds,
            fmt='none', ecolor=colors[i], capsize=4, alpha=0.9, linewidth=1.5
        )

        # Plot points and lines
        plt.plot(
            pulse_durations, thresholds,
            'o-', label=material_name, color=colors[i], alpha=0.9
        )
    else:
        print(f"Warning: File not found for {material_name}: {file_path}")


plt.xlabel("Pulse Duration [ps]", fontsize=14)
plt.ylabel("Threshold Energy [pJ]", fontsize=14)
plt.grid(False)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(output_path)
plt.show()
