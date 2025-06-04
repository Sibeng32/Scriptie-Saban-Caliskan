import os
import numpy as np
import scipy.ndimage as ndimage
import skimage.measure
import cv2

# Define input and output paths
data_dir = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/VoorSaban2/20250320/DataMainSetup"
save_dir = "/Users/sab/Desktop/Scriptie-Saban-Caliskan/Contrast_Method/Radii/20250320"
os.makedirs(save_dir, exist_ok=True)

# Define range of runs
first_run = 24
last_run = 55

def find_best_contrast_square(image, min_size=3, max_size=150, step=2):
    """Find the patch with highest contrast to global mean."""
    h, w = image.shape
    background_avg = np.mean(image)
    best_score = -np.inf
    best_params = (None, None, 0)

    for size in range(min_size, max_size + 1, step):
        half = size // 2
        for cx in range(half, h - half, step):
            for cy in range(half, w - half, step):
                patch = image[cx - half:cx + half, cy - half:cy + half]
                patch_avg = np.mean(patch)
                contrast = abs(patch_avg - background_avg)
                score = contrast * size**2
                if score > best_score:
                    best_score = score
                    best_params = (cx, cy, half)

    return best_params

def process_and_save_radii(run):
    """Process one run and save radii with coordinates using NumPy only."""
    filename = f"run_{str(run).zfill(3)}_overview.npy"
    filepath = os.path.join(data_dir, filename)
    data = np.load(filepath)
    radii_with_coords = np.zeros((64, 3))  # [radius, cx, cy]

    for idx in range(64):
        smoothed = ndimage.gaussian_filter(data[idx], 2)
        cx, cy, radius = find_best_contrast_square(smoothed)

        if cx is None or cy is None or radius == 0:
            continue

        patch = smoothed[cx - radius:cx + radius, cy - radius:cy + radius]
        patch_avg = np.mean(patch)
        global_avg = np.mean(smoothed)

        if abs(patch_avg - global_avg) < 0.01:
            continue

        radii_with_coords[idx] = [radius, cx, cy]

    save_path = os.path.join(save_dir, f"run_{str(run).zfill(3)}_radii.npy")
    np.save(save_path, radii_with_coords)
    print(f"Saved radii and coordinates to {save_path}")

# Run the batch
for run in range(first_run, last_run + 1):
    process_and_save_radii(run)
