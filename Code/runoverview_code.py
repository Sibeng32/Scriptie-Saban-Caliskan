
import numpy as np
import matplotlib.pyplot as plt
import os

# Run settings
first_run = 16
last_run  = 17
rows, cols = 8, 8
meander = False
out_dir  = "./overview"
os.makedirs(out_dir, exist_ok=True)

idx_x, idx_y = np.meshgrid(range(cols), range(rows))
bruh = r"a-C, $\tau = 0.2\ \mathrm{fs}$"

if meander:
    idx_x[::2, :] = idx_x[::2, ::-1]

idx_x = idx_x.flatten()
idx_y = idx_y.flatten()

for run in range(first_run, last_run):
    data = np.load(f'/Users/sab/Desktop/Scriptie-Saban-Caliskan/VoorSaban2/20250321/DataMainSetup/run_{str(run).zfill(3)}_overview.npy')    
    fig, ax = plt.subplots(rows, cols, figsize=(4.63, 5))
    
    for shot in range(rows * cols):
        img = data[shot]
        ax[idx_y[shot], idx_x[shot]].imshow(img, interpolation='none', cmap='gray')
        ax[idx_y[shot], idx_x[shot]].set_xticks([])
        ax[idx_y[shot], idx_x[shot]].set_yticks([])
    
    plt.subplots_adjust(left=0.01, right=.99, top=.92, bottom=0.01, wspace=0, hspace=0)

    fig.suptitle(bruh, fontsize=24, y=0.99)
    
    out_path = os.path.join(out_dir, f'{bruh}.pdf')
    print(f'printed {bruh}.pdf')
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
