import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

HW2_DIR = os.path.dirname(__file__)
SAVE_PATH = os.path.join(HW2_DIR, "all_images_img")

def find_files(hw2_dir):
    files = [f for f in os.listdir(hw2_dir) if f.endswith('.pt')]
    parsed = []
    for f in files:
        name = f[:-3]
        parts = name.split("_")
        if len(parts) != 4:
            continue
        c1, c2, trial, imgnum = parts
        parsed.append((int(c1), float("0."+c2), int(trial), int(imgnum), os.path.join(hw2_dir, f)))
    return parsed


def load_image(path):
    t = torch.load(path)
    arr = t.numpy()
    if arr.ndim == 3:
        # assume (C,H,W) -> convert to (H,W,C)
        arr = np.transpose(arr, (1,2,0))
    # if 2D, treat as grayscale HxW
    if arr.ndim == 2:
        arr = arr[..., None]
    # normalize per-image for display
    mi, ma = arr.min(), arr.max()
    if ma > mi:
        arr = (arr - mi) / (ma - mi)
    else:
        arr = np.zeros_like(arr)
    return arr


def make_grid(parsed, save_path, img_number=0, trials=(0,1,2), ):
    # build map (c1,c2) -> list of trial paths
    mapping = {}
    c1_vals = set()
    c2_vals = set()
    for c1, c2, trial, imgnum, path in parsed:
        if imgnum != img_number:
            continue
        c1_vals.add(c1)
        c2_vals.add(c2)
        mapping.setdefault((c1,c2), {})[trial] = path
    c1_list = sorted(list(c1_vals))
    c2_list = sorted(list(c2_vals))

    rows = len(c1_list)
    cols = len(c2_list)

    # Prepare figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]

    for i, c1 in enumerate(c1_list):
        for j, c2 in enumerate(c2_list):
            ax = axes[i, j]
            # collect images for each trial
            imgs = []
            for tr in trials:
                path = mapping.get((c1,c2), {}).get(tr)
                if path and os.path.exists(path):
                    imgs.append(load_image(path))
                else:
                    # placeholder blank image if missing
                    imgs.append(np.zeros((64,64,1), dtype=float))
            # ensure same height
            H = imgs[0].shape[0]
            # concatenate horizontally
            concat = np.concatenate([img if img.ndim==3 else img[...,None] for img in imgs], axis=1)
            if concat.shape[2] == 1:
                ax.imshow(concat[:,:,0], cmap='gray', vmin=0, vmax=1)
            else:
                # if >1 channels, clip to 3 channels for display
                img_disp = concat
                if img_disp.shape[2] > 3:
                    img_disp = img_disp[:, :, :3]
                ax.imshow(img_disp)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{c1},{c2}")

    plt.tight_layout()
    full_path=os.path.join(save_path, f"all_images_img{img_number}.png")
    fig.savefig(full_path, dpi=150)
    print(f"Saved combined figure to: {full_path}")


if __name__ == '__main__':
    HW2_DIR = os.path.dirname(__file__)
    SAVE_PATH = os.path.join(HW2_DIR, "all_images_img")
    generated_dir = '/home/zbalkhy/EEC289/hw2/content/generated'
    dirs = [d for d in os.listdir(generated_dir)] 
    for d in dirs:
        subdir = os.path.join(generated_dir,d)
        parsed = find_files(subdir)
        if len(parsed) == 0:
            print('No .pt files found in', subdir)
        else:
            make_grid(parsed, HW2_DIR, img_number=int(d))
