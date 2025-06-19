# -*- coding: utf-8 -*-
"""
Marnix Example
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops


first_run = 32
last_run = 48
rows, cols = 8, 8       # grid layout for plotting
pf = 0.8                # peak_fraction for threshold
gauss_sigma = 5         # sigma for Gaussian filter
meander = False         


data_dir = "../VoorSaban2/20250321/DataMainSetup"
out_dir = "./RadiiAndCoords/20250321"
os.makedirs(out_dir, exist_ok=True)

i_x, i_y = np.meshgrid(range(cols), range(rows))
if meander:
    i_x[::2, :] = i_x[::2, ::-1]
idx_x = i_x.flatten()
idx_y = i_y.flatten()


for run in range(first_run, last_run):
    fname = f"run_{run:03d}_overview.npy"
    stack = np.load(os.path.join(data_dir, fname))

    # Determine threshold from last frame
    last_img = gaussian_filter(stack[-1], sigma=gauss_sigma)
    thr = pf * (np.max(last_img) - 1) + 1

    # Prepare output array
    n_shots = stack.shape[0]
    radii_coords = np.zeros((n_shots, 3), dtype=float)

    fig, axes = plt.subplots(rows, cols, figsize=(4.63, 5))

    for shot in range(n_shots):
        img = stack[shot]
        g = gaussian_filter(img, sigma=gauss_sigma)
        bw = (g > thr).astype(int)
        lbl = label(bw)
        props = regionprops(lbl)
        if props:
            prop = max(props, key=lambda p: p.area)
            radius = prop.equivalent_diameter / 2.0
            cy, cx = prop.centroid
        else:
            radius, cx, cy = 0.0, 0.0, 0.0

        radii_coords[shot] = [radius, cx, cy]

        raw = (img - img.min()) / (img.max() - img.min())
        half = raw.shape[1] // 2
        loc = raw.copy()
        loc[:, half:] = ((g - g.min()) / (g.max() - g.min()))[:, half:] * 1.8
        ax = axes[idx_y[shot], idx_x[shot]]
        ax.imshow(loc, cmap='gray', interpolation='none')
        ax.contour(bw, levels=[0.5], colors='tab:orange', linewidths=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0, hspace=0)
    fig.savefig(os.path.join(out_dir, f"example_radii_run_{run:03d}.pdf"), transparent=True)
    plt.close(fig)

    # Save radii and coords array
    out_name = f"run_{run:03d}_radii_coords.npy"
    np.save(os.path.join(out_dir, out_name), radii_coords)
    print(f"Run {run:03d}: saved radii and coords to {out_name}")
