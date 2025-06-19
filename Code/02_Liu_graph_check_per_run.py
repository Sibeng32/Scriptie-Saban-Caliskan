import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.ndimage as ndimage

# Paths
radii_dir = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Radii/20250321"
image_dir = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/VoorSaban2/20250321/DataMainSetup"
first_run = 32
last_run = 47

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
    rows, cols = 8, 8
    sm = np.array([ndimage.gaussian_filter(img, 2) for img in data])

    # Index mapping for meander or regular scan (left-to-right, top-to-bottom)
    idx_x, idx_y = np.meshgrid(range(cols), range(rows))
    idx_x = idx_x.flatten()
    idx_y = idx_y.flatten()

    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(5, 5)  # Adjust as needed for your paper layout

    for i in range(len(sm)):
        r, cx, cy = radii_with_coords[i]
        ax = axs[idx_y[i], idx_x[i]]
        ax.imshow(sm[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        if r > 0:
            circle = plt.Circle((cy, cx), r, color='r', fill=False, linewidth=1)
            ax.add_patch(circle)

    # Tight layout with no space between subplots
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0, hspace=0)
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
