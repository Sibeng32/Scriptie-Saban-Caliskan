import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.ndimage as ndimage

# Paths
radii_dir = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Radii/20250320"
image_dir = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/VoorSaban2/20250321/DataMainSetup"
first_run = 24
last_run = 55

output_pdf = os.path.join(radii_dir, f"liu_and_circles_run_{first_run}-{last_run}.pdf")

def plot_liu_from_radii(radii_with_coords, run):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Liu Plot for Run {run}", fontsize=14, fontweight='bold')

    radii = radii_with_coords[:, 0]
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

def plot_circles_on_images(data, radii_with_coords, run):
    smoothed_data = np.array([ndimage.gaussian_filter(img, 2) for img in data])

    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle(f"Detected Circles for Run {run}", fontsize=16, fontweight='bold')

    for idx in range(64):
        ax = axs[idx // 8, idx % 8]
        ax.imshow(smoothed_data[idx], cmap='gray')
        ax.axis('off')

        radius, cx, cy = radii_with_coords[idx]
        if radius == 0:
            continue

        circle = plt.Circle((cy, cx), radius, color='r', fill=False, linewidth=1)
        ax.add_patch(circle)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(run)
    return fig

# Generate PDF
with PdfPages(output_pdf) as pdf:
    for run in range(first_run, last_run + 1):
        radii_path = os.path.join(radii_dir, f"run_{str(run).zfill(3)}_radii.npy")
        image_path = os.path.join(image_dir, f"run_{str(run).zfill(3)}_overview.npy")

        if not os.path.exists(radii_path) or not os.path.exists(image_path):
            print(f"Skipping run {run}: missing file(s)")
            continue

        radii_data = np.load(radii_path)
        image_data = np.load(image_path)

        liu_fig = plot_liu_from_radii(radii_data, run)
        pdf.savefig(liu_fig)
        plt.close(liu_fig)

        circle_fig = plot_circles_on_images(image_data, radii_data, run)
        pdf.savefig(circle_fig)
        plt.close(circle_fig)

print(f"Liu + circle PDF saved to: {output_pdf}")
