import numpy as np
import scipy.ndimage as ndimage
import skimage.measure
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Run settings
first_run = 24
last_run = 55
rows, cols = 8, 8

data_dir = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/VoorSaban2/20250320/DataMainSetup"
output_pdf = os.path.join(
    data_dir, f"all_runs_overview_with_fitted_circle_and_Liu_plot_{first_run}-{last_run}_max_contrast_method.pdf"
)

# This function finds the square region with the highest contrast average difference to the average of the entire array.
# loop over sizes and positions to find the best combination of central coordinates and radius

def find_best_contrast_square(image, min_size=3, max_size=150, step=2):
    h, w = image.shape
    background_avg = np.mean(image)
    best_score = -np.inf # start with lowest possible score
    best_params = (None, None, 0)
    
    #loop over all sizes
    for size in range(min_size, max_size + 1, step):
        half = size // 2
        #loop over all possible x and y centre coordinates
        for cx in range(half, h - half, step):
            for cy in range(half, w - half, step):
                patch = image[cx - half:cx + half, cy - half:cy + half]
                patch_avg = np.mean(patch)
                contrast = abs(patch_avg - background_avg)
                score = contrast * (size ** 2)
                
                #Save if new score is best found
                if score > best_score:
                    best_score = score
                    best_params = (cx, cy, half)

    return best_params

# Process all 64 numpy arrays in a run
# For each one: smooth with Gaussian, find best circle, threshold with contrast cutoff

def process_run(run):
    filename = f"run_{str(run).zfill(3)}_overview.npy"
    filepath = os.path.join(data_dir, filename)
    print(f"Processing {filename}...")

    data = np.load(filepath)
    thresh = np.zeros(data.shape, dtype=bool)
    thresh_pos = np.zeros(data.shape, dtype=bool) # I don't do anything with this, but too lazy to remove it and change the entire code
    circles_list = [[] for _ in range(64)]
    circles_pos_list = [[] for _ in range(64)]

    for idx in range(64):
        image = data[idx]
        smoothed = ndimage.gaussian_filter(image, 2) #Gaussian Filter, sigma = 2

        cx, cy, radius = find_best_contrast_square(smoothed)

        if cx is None or cy is None or radius == 0:
            continue

        # Make sure the found region actually is different from the background
        patch = smoothed[cx - radius:cx + radius, cy - radius:cy + radius]
        patch_avg = np.mean(patch)
        global_avg = np.mean(smoothed)

        # This to cut off plotting circles in noise or needlesly keep going on. 
        if abs(patch_avg - global_avg) < 0.01:   
            continue

        mask = np.zeros_like(smoothed, dtype=np.uint8)
        cv2.circle(mask, (cy, cx), radius, 1, thickness=-1)

        thresh[idx] = mask.astype(bool)
        thresh_pos[idx] = np.zeros_like(mask, dtype=bool)
        circles_list[idx] = [(cx, cy, radius)]
        circles_pos_list[idx] = []

    np.save(os.path.join(data_dir, f"run_{str(run).zfill(3)}_thresh.npy"), thresh)
    
    # I don't do anything with this, but too lazy to remove it and change the entire code
    np.save(os.path.join(data_dir, f"run_{str(run).zfill(3)}_thresh_pos.npy"), thresh_pos) 

    return data, thresh, thresh_pos, circles_list, circles_pos_list

# Plots radius squared vs log index and fits a line for all non-zero data
def plot_liu(biggest_circles):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Liu Plot: Radius² vs Log(Index) + Linear Fit", fontsize=14)

    radii_squared = [r**2 if r else 0 for _, _, r in biggest_circles]
    x_vals = np.arange(1, len(radii_squared) + 1)
    nonzero_x = [i for i, r in zip(x_vals, radii_squared) if r > 0]
    nonzero_y = [r for r in radii_squared if r > 0]

    if len(nonzero_x) > 1:
        log_x = np.log(nonzero_x)
        a, b = np.polyfit(log_x, nonzero_y, 1)
        ax.plot(x_vals, a * np.log(x_vals) + b, linestyle="--", color="red",
                label=f"fit: y = {a:.2f} log(x) + {b:.2f}")

    scatter = ax.scatter(x_vals, radii_squared, c=radii_squared, cmap="viridis", edgecolors='k')
    ax.set_xscale("log")
    ax.set_xlabel("Log(Index)")
    ax.set_ylabel("Radius²")
    ax.legend()
    ax.grid(True)
    plt.colorbar(scatter, ax=ax, label="Radius²")
    return fig

# Makes a PDF pages with the visualizations for all 64 images
def save_pdf(runs_data):
    with PdfPages(output_pdf) as pdf:
        for run, (data, thresh, thresh_pos, circles_list, circles_pos_list) in runs_data.items():
            fig, axes = plt.subplots(8, 8, figsize=(16, 16))
            fig.suptitle(f"Run {run} Overview") #

            biggest_circles = [None] * 64
            for idx in range(64):
                r = idx // 8
                c = idx % 8
                ax = axes[r, c]

                ax.imshow(data[idx], cmap="gray")
                ax.imshow(thresh[idx], cmap="Blues", alpha=0.6)

                circle = max(circles_list[idx], key=lambda c: c[2], default=None)
                if circle:
                    x, y, rad = circle
                    ax.add_patch(plt.Circle((y, x), rad, color='red', fill=False, linewidth=2))
                    biggest_circles[idx] = circle
                else:
                    biggest_circles[idx] = (0, 0, 0)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Idx {idx}", fontsize=8)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Add Liu plot page
            fig2 = plot_liu(biggest_circles)
            pdf.savefig(fig2) #gebruik fig.adjust subplots. vspace and wspace
            plt.close(fig2)

    print(f"PDF saved to: {output_pdf}")

# Run the code Bombaclat
runs_data = {}
for run in range(first_run, last_run + 1):
    results = process_run(run)
    runs_data[run] = results

save_pdf(runs_data)
