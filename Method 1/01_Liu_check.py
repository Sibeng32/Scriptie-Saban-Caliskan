# -*- coding: utf-8 -*-
"""
Generate Liu plots and circle overlays using precomputed radii & centroid arrays,
combining figures into a PDF without re-extracting radii.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

# --- User settings ---------------------------------------------------------
first_run = 32
last_run = 47   
rows, cols = 8, 8        # grid layout


# Paths (adjust as needed)
radii_dir = "./RadiiAndCoords/20250321"
data_dir = "../VoorSaban2/20250321/DataMainSetup"
out_pdf = os.path.join(radii_dir, f"liu_and_circles_run_{first_run:02d}-{last_run:02d}_M1.pdf")

# Subplot index mapping
i_x, i_y = np.meshgrid(range(cols), range(rows))
idx_x = i_x.flatten()
idx_y = i_y.flatten()

# Function: create Liu plot from existing radii array
def plot_liu(rc, run):
    radii_sq = rc[:,0]**2
    x = np.arange(1, len(radii_sq)+1)
    nonzero = radii_sq > 0

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x, radii_sq, c=radii_sq, cmap='viridis', edgecolors='black')
    ax.set_xscale('log')
    ax.set_xlabel('Log(Index)')
    ax.set_ylabel('Radius²')
    ax.set_title(f'Liu Plot Run {run:03d}')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    if nonzero.sum()>1:
        a, b = np.polyfit(np.log(x[nonzero]), radii_sq[nonzero], 1)
        ax.plot(x, a*np.log(x) + b, 'r--', label=f'y = {a:.2f} ln(x) + {b:.2f}')
        ax.legend()
    ax.set_ylim(bottom=0)
    return fig

# Function: overlay circles using existing radii & centroid data
def plot_circles(stack, rc, run):
    sm = np.array([gaussian_filter(img, sigma=2) for img in stack])
    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(5, 5)  # Match your preferred aspect

    for i in range(rc.shape[0]):
        r, cx, cy = rc[i]
        ax = axs[idx_y[i], idx_x[i]]
        ax.imshow(sm[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        if r > 0:
            circ = plt.Circle((cx, cy), r, color='r', fill=False, linewidth=1)
            ax.add_patch(circ)

    plt.subplots_adjust(left=0.01, right=.99, top=.99, bottom=0.01, wspace=0, hspace=0)
    return fig

# Main: load existing radii & images, build PDF
with PdfPages(out_pdf) as pdf:
    for run in range(first_run, last_run + 1):
        radii_path = os.path.join(radii_dir, f"run_{run:03d}_radii_coords.npy")
        image_path = os.path.join(data_dir, f"run_{run:03d}_overview.npy")
        if not os.path.exists(radii_path) or not os.path.exists(image_path):
            print(f"Skipping run {run:03d}: missing files.")
            continue

        rc = np.load(radii_path)
        stack = np.load(image_path)

        # Liu plot
        liu_fig = plot_liu(rc, run)
        pdf.savefig(liu_fig)
        # plt.subplots_adjust(left=0.01, right=.99, top=.92, bottom=0.01, wspace=0, hspace=0)
        plt.close(liu_fig)

        # Circles overlay
        circ_fig = plot_circles(stack, rc, run)
        pdf.savefig(circ_fig)
        plt.close(circ_fig)

print(f"Saved Liu & circle PDF to: {out_pdf}")
