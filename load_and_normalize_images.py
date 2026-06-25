"""
Load images from folder, normalize by group (D1-D5) and band (EL, UV, VI),
and display before/after normalization examples.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import re
from tqdm import tqdm

# Configuration for both folders

import re
import argparse

BANDS = ['EL', 'UV', 'VI']

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def get_groups_from_filenames(fpath):
    print(f"\nDetecting groups from filenames in: {fpath}")
    panel = os.path.basename(fpath)
    panel2 = panel[:-2] + '_' + panel[-1]
    #print(f"Panel: {panel}")
    groups = set()
    files = [f for f in os.listdir(fpath) if f.endswith(".tif")]
    for fname in files:
        match = re.search(rf"{re.escape(panel)}(\d+)", fname)
        match2 = re.search(rf"{re.escape(panel2)}(\d+)", fname)
        if match:
                no = match.group(1)
        elif match2:
                no = match2.group(1)
        groups.add(f"{panel[-1]}{no}")
    return sorted(list(groups))


def load_images_by_group_and_band(folder_path, panel, groups):
    """
    Load images organized by group and band.
    Returns: dict[group][band] = list of (filename, image_array)
    """
    # Accept groups as argument
    # Detect group prefix from folder name
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Determine group prefix and group list
    '''
    if "23-P09-D" in str(folder_path):
        flag = 'D'
        pattern = re.compile(r'23-P09-D(\d)_(EL|UV|VI)_Cell\d+', re.IGNORECASE)
    elif "24-P10-A" in str(folder_path):
        flag = 'A'
        pattern = re.compile(r'24-P10_A(\d)_(EL|UV|VI)_Cell\d+', re.IGNORECASE)
    elif "25-019-A" in str(folder_path):
        flag = 'A'
        pattern = re.compile(r'25-019_A(\d)_(EL|UV|VI)_Cell\d+', re.IGNORECASE)
    else:
        raise ValueError(f"Unknown folder structure: {folder_path}")
    '''
    flag = panel[-1]  # Extract 'B' from '23-P09-B'
    panel2 = panel[:-2] + '_' + panel[-1]  # '24-128-A' -> '24-128_A'

    pattern = re.compile(rf'{re.escape(panel)}(\d)_(EL|UV|VI)_Cell\d+', re.IGNORECASE)
    pattern2 = re.compile(rf'{re.escape(panel2)}(\d)_(EL|UV|VI)_Cell\d+', re.IGNORECASE)
    #print(f"Using regex pattern:  {pattern.pattern}")
    #print(f"Using regex pattern2: {pattern2.pattern}")
    images = {group: {band: [] for band in BANDS} for group in groups}

    
    
    all_files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    with tqdm(total=len(all_files), desc="Loading images", unit="file") as pbar:
        for img_file in all_files:
            match = pattern.search(img_file.stem) or pattern2.search(img_file.stem)
            #print(f"Processing file: {img_file.name}, Match: {match.group(0) if match else 'No match'}")
            if match:
                group = f'{flag}{match.group(1)}'
                band = match.group(2).upper()
                print(f"Detected group: {group}, band: {band}")
                if group in images and band in BANDS:
                    for attempt in range(3):
                        try:
                            with Image.open(img_file) as img:
                                img_array = np.array(img, dtype=np.float32)
                                #print(f"Loaded image: {img_file.name}, shape: {img_array.shape}, dtype: {img_array.dtype}")
                            images[group][band].append((img_file.name, img_array))
                            break
                        except (TimeoutError, OSError) as e:
                            if attempt < 2:
                                print(f"\nRetry {attempt+1}/3 for {img_file.name}: {e}")
                            else:
                                print(f"\nSkipping {img_file.name} after 3 failed attempts: {e}")
            pbar.update(1)
    return images


def compute_min_and_topk_percentile(image_arrays, k=100, percentile=30):
    """
    Compute global min and percentile over top-k brightest pixels across images
    without concatenating all pixels into one huge array.
    """
    global_min = np.inf
    topk = np.array([], dtype=np.float32)

    for img in image_arrays:
        flat = img.ravel()
        if flat.size == 0:
            continue

        img_min = float(flat.min())
        if img_min < global_min:
            global_min = img_min

        if flat.size > k:
            local_topk = np.partition(flat, -k)[-k:]
        else:
            local_topk = flat

        if topk.size == 0:
            topk = local_topk.astype(np.float32, copy=False)
        else:
            merged = np.concatenate((topk, local_topk.astype(np.float32, copy=False)))
            if merged.size > k:
                topk = np.partition(merged, -k)[-k:]
            else:
                topk = merged

    if not np.isfinite(global_min):
        return 0.0, 0.0

    if topk.size == 0:
        return float(global_min), float(global_min)

    max_val = float(np.percentile(topk, percentile))
    return float(global_min), max_val

def plot_original_brightest_el(images, output_file='el_brightest_darkest_original.png'):
    print("\nBrightest and darkest ORIGINAL EL images per group:")
    import matplotlib.pyplot as plt
    n_groups = len(images.keys())
    fig, axes = plt.subplots(n_groups, 2, figsize=(8, n_groups * 3))
    fig.suptitle('Original EL Images: Brightest (left) vs Darkest (right)', fontsize=14)
    for i, group in enumerate(images.keys()):
        el_images = images[group]['EL']
        if el_images:
            # Find brightest and darkest by mean pixel value
            means = [img.max() for _, img in el_images]
            brightest_idx = int(np.argmax(means))
            darkest_idx = int(np.argmin(means))
            bright_name, bright_img = el_images[brightest_idx]
            dark_name, dark_img = el_images[darkest_idx]
            # Brightest
            ax_bright = axes[i,0] if n_groups > 1 else axes[0]
            ax_bright.imshow(bright_img.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
            ax_bright.set_title(f'{group} EL Brightest (Original)\n{bright_name}\nmean={bright_img.mean():.1f}', fontsize=9)
            ax_bright.axis('off')
            # Darkest
            ax_dark = axes[i,1] if n_groups > 1 else axes[1]
            ax_dark.imshow(dark_img.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
            ax_dark.set_title(f'{group} EL Darkest (Original)\n{dark_name}\nmean={dark_img.mean():.1f}', fontsize=9)
            ax_dark.axis('off')
        else:
            # No EL images
            ax_bright = axes[i,0] if n_groups > 1 else axes[0]
            ax_dark = axes[i,1] if n_groups > 1 else axes[1]
            ax_bright.text(0.5, 0.5, f'{group} EL\nNo image', ha='center', va='center', fontsize=12)
            ax_dark.text(0.5, 0.5, f'{group} EL\nNo image', ha='center', va='center', fontsize=12)
            ax_bright.axis('off')
            ax_dark.axis('off')
    plt.tight_layout()
    # Remove saving and printing here; handled in main()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
 


def normalize_images(images):
    """
    Normalize images to 0-255 range by group and band.
    Returns: dict[group][band] = list of (filename, normalized_array, original_array)
    """
    normalized = {group: {band: [] for band in BANDS} for group in images.keys()}
    for group in images.keys():
        for band in BANDS:
            if not images[group][band]:
                continue
            
            # Collect all images in this group-band combination
            all_images = [img for _, img in images[group][band]]
            
            if not all_images:
                continue
            
            # Compute robust normalization stats without global full-array sort.
            min_val, max_val = compute_min_and_topk_percentile(all_images, k=100, percentile=30)
            print(f"\n{group}-{band}: min={min_val:.2f}, max={max_val:.2f} (using 0.-99.99 percentiles)")
            # Normalize each image using group-wide min and max

            for filename, img in images[group][band]:
                if max_val > 254:
                    normalized_img = img.astype(np.uint8)
                else:
                    if max_val > min_val:
                        normalized_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        normalized_img = np.zeros_like(img, dtype=np.uint8)

                normalized[group][band].append((filename, normalized_img, img))
    return normalized


def display_comparison(normalized_images, output_file='normalization_comparison.png'):
    """
    Display before/after normalization for one example from each group and band.
    """
    n_groups = len(normalized_images.keys())
    n_bands = len(BANDS)
    
    fig, axes = plt.subplots(n_groups, n_bands * 2, figsize=(n_bands * 6, n_groups * 3))
    fig.suptitle('Image Normalization: Before (left) vs After (right)', fontsize=16, y=0.995)
    
    from tqdm import tqdm
    total_steps = len(normalized_images.keys()) * len(BANDS)
    with tqdm(total=total_steps, desc="Creating visualization", unit="cell") as pbar:
        for i, group in enumerate(normalized_images.keys()):
            for j, band in enumerate(BANDS):
                # Get first image from this group-band
                if normalized_images[group][band]:
                    filename, norm_img, orig_img = normalized_images[group][band][0]
                    # Before normalization
                    ax_before = axes[i, j * 2] if n_groups > 1 else axes[j * 2]
                    #im_before = ax_before.imshow(orig_img, cmap='gray')
                    orig_uint8 = orig_img.astype(np.uint8)
                    im_before = ax_before.imshow(orig_uint8, cmap='gray', vmin=0, vmax=255)
                    ax_before.set_title(f'{group}-{band}\nBefore\n{filename}', fontsize=10)
                    ax_before.axis('off')
                    plt.colorbar(im_before, ax=ax_before, fraction=0.046, pad=0.04)
                    # After normalization
                    ax_after = axes[i, j * 2 + 1] if n_groups > 1 else axes[j * 2 + 1]
                    im_after = ax_after.imshow(norm_img, cmap='gray', vmin=0, vmax=255)
                    ax_after.set_title(f'{group}-{band}\nAfter (0-255)\n{filename}', fontsize=10)
                    ax_after.axis('off')
                    plt.colorbar(im_after, ax=ax_after, fraction=0.046, pad=0.04)
                else:
                    # No image found
                    ax_before = axes[i, j * 2] if n_groups > 1 else axes[j * 2]
                    ax_after = axes[i, j * 2 + 1] if n_groups > 1 else axes[j * 2 + 1]
                    ax_before.text(0.5, 0.5, f'{group}-{band}\nNo image', 
                                  ha='center', va='center', fontsize=12)
                    ax_after.text(0.5, 0.5, f'{group}-{band}\nNo image', 
                                 ha='center', va='center', fontsize=12)
                    ax_before.axis('off')
                    ax_after.axis('off')
                pbar.update(1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to: {output_file}")
    # plt.show() removed for faster execution


def save_normalized_images(normalized_images, output_folder='normalized_images'):
    """
    Save normalized images to output folder.
    """
    output_path = Path(output_folder)
    if not output_path.exists():
        output_path.mkdir()

    band_folders = {}
    for band in BANDS:
        out_folder = output_path / band
        out_folder.mkdir(exist_ok=True)
        band_folders[band] = out_folder

    for group in normalized_images.keys():
        for band in BANDS:
            for filename, norm_img, _ in normalized_images[group][band]:
                out_folder = band_folders[band]
                output_file =  out_folder / f"{Path(filename).stem}_normalized{Path(filename).suffix}"
                if band == 'EL':
                    # Ensure EL image is 2D before saving as grayscale
                    norm_img_uint8 = norm_img.astype(np.uint8)
                    if norm_img_uint8.ndim == 3:
                        # If 3D, take first channel (assume grayscale)
                        norm_img_uint8 = norm_img_uint8[..., 0]
                    Image.fromarray(norm_img_uint8, mode='L').save(output_file)
                else:
                    Image.fromarray(norm_img).save(output_file)
                print(f"Saved: {output_file}")


def plot_normalized_brightest_el(normalized_images, output_file='el_brightest_darkest.png'):
    # Show brightest and darkest EL image from each group after normalization
    print("\nBrightest and darkest EL images per group (after normalization):")
    import matplotlib.pyplot as plt
    n_groups = len(normalized_images.keys())
    fig, axes = plt.subplots(n_groups, 2, figsize=(8, n_groups * 3))
    for i, group in enumerate(normalized_images.keys()):
        el_images = normalized_images[group]['EL']
        if el_images:
            # Find brightest and darkest by mean pixel value
            means = [img.mean() for _, img, _ in el_images]
            brightest_idx = int(np.argmax(means))
            darkest_idx = int(np.argmin(means))
            bright_name, bright_img, _ = el_images[brightest_idx]
            dark_name, dark_img, _ = el_images[darkest_idx]
            # Print min/max for D5 EL
            min_val = bright_img.min()
            max_val = bright_img.max()
            print(f"D5 EL normalized brightest image: {bright_name}")
            print(f"normalized min: {min_val:.2f}, max: {max_val:.2f}, mean: {bright_img.mean():.2f}")
            # Brightest
            ax_bright = axes[i,0] if n_groups > 1 else axes[0]
            ax_bright.imshow(bright_img, cmap='gray', vmin=0, vmax=255)
            ax_bright.set_title(f'{group} EL Brightest\n{bright_name}', fontsize=10)
            ax_bright.axis('off')
            # Darkest
            ax_dark = axes[i,1] if n_groups > 1 else axes[1]
            ax_dark.imshow(dark_img, cmap='gray', vmin=0, vmax=255)
            ax_dark.set_title(f'{group} EL Darkest\n{dark_name}', fontsize=10)
            ax_dark.axis('off')
        else:
            # No EL images
            ax_bright = axes[i,0] if n_groups > 1 else axes[0]
            ax_dark = axes[i,1] if n_groups > 1 else axes[1]
            ax_bright.text(0.5, 0.5, f'{group} EL\nNo image', ha='center', va='center', fontsize=12)
            ax_dark.text(0.5, 0.5, f'{group} EL\nNo image', ha='center', va='center', fontsize=12)
            ax_bright.axis('off')
            ax_dark.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="24-P10-A", type=str, help="Panel name, e.g. 23-P09-A")
    args = parser.parse_args()
    panel = args.panel
    one_drive_path = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage/bboxes/"
    fpath = os.path.join(one_drive_path, panel)
    groups = get_groups_from_filenames(fpath.rstrip('/'))
    print(f"Detected groups in {panel}: {groups}")

    IMAGE_FOLDERS = [
    #{"path": "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage/bboxes/23-P09-D/", "groups": ['D1', 'D2', 'D3', 'D4', 'D5']},
    #{"path": "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage/bboxes/24-P10-A/", "groups": ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']},
    #{"path": "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage/bboxes/25-019-A/", "groups": ['A1', 'A2']}
    {"path": fpath, "groups": groups  }

    ]
    

    for folder_cfg in IMAGE_FOLDERS:
        image_folder = folder_cfg["path"]
        folder_name = Path(image_folder.rstrip('/')).name
        # Create output subfolders for this dataset
        grid_dir = Path("grid") / folder_name
        grid_dir.mkdir(parents=True, exist_ok=True)
        norm_dir = Path("normalized_images") / folder_name
        norm_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[1/4] Loading images from {image_folder} ...")
        images = load_images_by_group_and_band(image_folder, panel, folder_cfg["groups"])
        # Count loaded images
        total = sum(len(images[g][b]) for g in images.keys() for b in BANDS)
        print(f"\nTotal images loaded: {total}")
        # Save original brightest/darkest plot
        plot_original_brightest_el(images, grid_dir / 'el_brightest_darkest_original.png')
        # Normalize images
        print("\n[2/4] Normalizing images by group and band...")
        normalized_images = normalize_images(images)
        print("="*80)
        print(f"IMAGE NORMALIZATION BY GROUP AND BAND for {image_folder}")
        print("="*80)
        # Display comparison
        print("\n[3/4] Creating visualization...")
        display_comparison(normalized_images, grid_dir / 'normalization_comparison.png')
        plot_normalized_brightest_el(normalized_images, grid_dir / 'el_brightest_darkest.png')
        # Save normalized images
        print("\n[4/4] Saving normalized images...")
        save_normalized_images(normalized_images, output_folder=norm_dir)
        for group in normalized_images.keys():
            for band in BANDS:
                n_images = len(normalized_images[group][band])
                print(f"Group {group}, Band {band}: {n_images} normalized images saved.")
                # Plot all images in a grid and save as JPG
                specific_images = normalized_images[group][band]
                # Sort by cell number extracted from filename
                def extract_cell_number(fname):
                    match = re.search(r'Cell(\d+)', fname)
                    return int(match.group(1)) if match else -1
                specific_images_sorted = sorted(specific_images, key=lambda x: extract_cell_number(x[0]))
                n_images = len(specific_images_sorted)
                n_rows, n_cols = 5, 8
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
                for idx, (fname, norm_img, orig_img) in enumerate(specific_images_sorted):
                    if idx >= n_rows * n_cols:
                        break
                    row = idx // n_cols
                    col = idx % n_cols
                    ax = axes[row, col]
                    ax.imshow(norm_img, cmap='gray')
                    # Extract cell number from filename using regex
                    match = re.search(r'Cell(\d+)', fname)
                    cell_number = match.group(1) if match else ''
                    ax.set_title(f'Cell{cell_number}', fontsize=7)
                    ax.axis('off')
                # Hide unused axes if fewer than 40 images
                for idx in range(n_images, n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].axis('off')
                plt.tight_layout()
                plt.savefig(grid_dir / f'{group}_{band}_grid.jpg', dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved: {grid_dir}/{group}_{band}_grid.jpg')


if __name__ == '__main__':
    main()
