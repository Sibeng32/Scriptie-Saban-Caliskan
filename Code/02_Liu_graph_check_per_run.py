import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

data_dir = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Radii/20250320" 
first_run = 24
last_run = 55

output_pdf = os.path.join(data_dir, f"liu_plots_from_radii_run_{first_run}-{last_run}.pdf")

def plot_liu_from_radii(radii, run):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Liu Plot for Run {run}", fontsize=14, fontweight='bold')

    radii_squared = radii ** 2
    x_values = np.arange(1, len(radii_squared) + 1)

    nonzero_indices = [i for i, r in zip(x_values, radii_squared) if r > 0]
    nonzero_radii = [r for r in radii_squared if r > 0]

    if len(nonzero_indices) > 1:
        log_x = np.log(nonzero_indices)
        a, b = np.polyfit(log_x, nonzero_radii, 1)
        fitted_y = a * np.log(x_values) + b
        ax.plot(x_values, fitted_y, 'r--', label=f"y = {a:.2f} log(x) + {b:.2f}")

    scatter = ax.scatter(x_values, radii_squared, c=radii_squared, cmap='viridis', edgecolors='black')
    ax.set_xscale('log')
    ax.set_xlabel("Log(Index)", fontsize=12)
    ax.set_ylabel("Radius²", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Radius²")
    ax.set_ylim(bottom=0)


    return fig

# PDF generation
with PdfPages(output_pdf) as pdf:
    for run in range(first_run, last_run + 1):
        filepath = os.path.join(data_dir, f"run_{str(run).zfill(3)}_radii.npy")
        if not os.path.exists(filepath):
            print(f"Skipping run {run}: radii file not found")
            continue

        radii = np.load(filepath)
        fig = plot_liu_from_radii(radii, run)
        pdf.savefig(fig)
        plt.close(fig)

print(f"Liu PDF saved to: {output_pdf}")
