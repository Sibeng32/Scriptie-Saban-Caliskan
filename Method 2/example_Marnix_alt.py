# -*- coding: utf-8 -*-
"""
Contrast-based crater detection using shadow casting, with radius extraction.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops

def cast_shadow_ud(img, ct_thres):
    shadow = np.zeros_like(img)
    for i, row in enumerate(img[1:]):
        shadow[i+1] = np.max(np.array([shadow[i], row < ct_thres]), axis=0)
    return shadow

def cast_shadow_du(img, ct_thres):
    shadow = np.zeros_like(img)
    for i, row in enumerate(np.flipud(img)[1:]):
        shadow[i+1] = np.max(np.array([shadow[i], row < ct_thres]), axis=0)
    return np.flipud(shadow)

def crater_area(img, ct_thres):
    return cast_shadow_ud(img, ct_thres) == (cast_shadow_du(img, ct_thres) + 1) / 2

def get_contrast_threshold_from_background(run, factor=2.5):
    corners = np.array([
        run[:, :10, :10],      # top-left
        run[:, -10:, :10],     # bottom-left
        run[:, -10:, -10:],    # bottom-right
        run[:, :10, -10:]      # top-right
    ])
    contrast_threshold_low = np.median(corners) - factor * np.std(corners)
    contrast_threshold_high = np.median(corners) + factor * np.std(corners)
    return contrast_threshold_low, contrast_threshold_high

first_run = 2
last_run = 40
rows, cols = 8, 8
meander = False
sigma = 5
contrast_factor = 0.3

data_dir = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/VoorSaban2/20250320/DataMainSetup"
out_dir = "/Users/sab/Documents/GitHub/Scriptie-Saban-Caliskan/Method 2/20250320"
os.makedirs(out_dir, exist_ok=True)


i_x, i_y = np.meshgrid(range(cols), range(rows))
if meander:
    i_x[::2, :] = i_x[::2, ::-1]
idx_x = i_x.flatten()
idx_y = i_y.flatten()


for run in range(first_run, last_run):
    fname = f"run_{run:03d}_overview.npy"
    data = np.load(os.path.join(data_dir, fname))
    contrast_threshold_low, _ = get_contrast_threshold_from_background(data, factor=contrast_factor)

    fig, axes = plt.subplots(rows, cols, figsize=(4.63, 5))
    radii_coords = np.zeros((rows * cols, 3), dtype=float)

    for shot in range(rows * cols):
        img = data[shot]
        fimg = gaussian_filter(img, sigma=sigma)
        bw = crater_area(fimg, contrast_threshold_low).astype(int)
        lbl = label(bw)
        props = regionprops(lbl)
        if props:
            prop = max(props, key=lambda p: p.area)
            radius = prop.equivalent_diameter / 2.0
            cy, cx = prop.centroid
        else:
            radius, cx, cy = 0.0, 0.0, 0.0

        radii_coords[shot] = [radius, cx, cy]

        # Combined image with enhanced filtered half
        raw = (img - img.min()) / (img.max() - img.min())
        filt = (fimg - fimg.min()) / (fimg.max() - fimg.min())
        loc = raw.copy()
        loc[:, img.shape[1] // 2:] = filt[:, img.shape[1] // 2:] * 1.8

        ax = axes[idx_y[shot], idx_x[shot]]
        ax.imshow(loc, cmap='gray', interpolation='none')
        ax.contour(bw, levels=[0.5], colors='tab:orange', linewidths=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0, hspace=0)
    fig.savefig(os.path.join(out_dir, f"contrast_radii_run_{run:03d}.pdf"), transparent=True)
    plt.close(fig)

    np.save(os.path.join(out_dir, f"run_{run:03d}_radii_coords.npy"), radii_coords)
    print(f"Run {run:03d}: saved radii and coords.")
