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


# Configuration for both folders
IMAGE_FOLDERS = [
 #   {"path": "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage/bboxes/23-P09-D/", "groups": ['D1', 'D2', 'D3', 'D4', 'D5']},
    {"path": "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage/bboxes/24-P10-A/", "groups": ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']}
]
BANDS = ['EL', 'UV', 'VI']


def load_images_by_group_and_band(folder_path):
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
    if "23-P09-D" in str(folder_path):
        groups = ['D1', 'D2', 'D3', 'D4', 'D5']
        pattern = re.compile(r'23-P09-D(\d)_(EL|UV|VI)_Cell\d+', re.IGNORECASE)
    elif "24-P10-A" in str(folder_path):
        groups = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        pattern = re.compile(r'24-P10-A(A\d)_(EL|UV|VI)_Cell\d+', re.IGNORECASE)
    else:
        raise ValueError(f"Unknown folder structure: {folder_path}")
    images = {group: {band: [] for band in BANDS} for group in groups}

    from tqdm import tqdm
    all_files = list(folder.glob('**/*'))
    matched_files = []
    with tqdm(total=len(all_files), desc="Loading images", unit="file") as pbar:
        for img_file in all_files:
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                match = pattern.search(img_file.stem)
                if match:
                    group = f'D{match.group(1)}'
                    band = match.group(2).upper()
                    if group in images and band in BANDS:
                        img = Image.open(img_file)
                        img_array = np.array(img, dtype=np.float32)
                        print(img_array.min(), img_array.max(), img_file.name)

                        images[group][band].append((img_file.name, img_array))
                        matched_files.append(f"Loaded: {img_file.name} -> Group: {group}, Band: {band}")
            pbar.update(1)
    print("\nMatched files:")
    for f in matched_files:
        print(f)
    return images


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
            
            # Flatten all images to find group-wide min and max for this group-band
            all_pixels = np.concatenate([img.flatten() for img in all_images])
            # Use percentiles instead of absolute min/max to ignore outlier bright pixels
            min_val = all_pixels.min()
            last_pixels = (np.sort(all_pixels)[-100:])
            print(last_pixels)
            max_val = np.percentile(last_pixels, 30)  # 50th percentile for max brightest 100 pixels
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
                    im_before = ax_before.imshow(orig_img, cmap='gray')
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
    print(f"\nComparison plot saved to: {output_file}")
    # plt.show() removed for faster execution


def save_normalized_images(normalized_images, output_folder='normalized_images'):
    """
    Save normalized images to output folder.
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    for group in normalized_images.keys():
        for band in BANDS:
            for filename, norm_img, _ in normalized_images[group][band]:
                output_file = output_path / f"{Path(filename).stem}_normalized{Path(filename).suffix}"
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


def plot_normalized_brightest_el(normalized_images):

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
            plt.savefig('el_brightest_darkest.png', dpi=150, bbox_inches='tight')
            print("Saved: el_brightest_darkest.png")



def plot_original_brightest_el(images):
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
    
def main():


    for folder_cfg in IMAGE_FOLDERS:
        image_folder = folder_cfg["path"]
        folder_name = Path(image_folder.rstrip('/')).name
        # Create output subfolders for this dataset
        grid_dir = Path("grid") / folder_name
        grid_dir.mkdir(parents=True, exist_ok=True)
        norm_dir = Path("normalized_images") / folder_name
        norm_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[1/4] Loading images from {image_folder} ...")
        images = load_images_by_group_and_band(image_folder)
        # Count loaded images
        total = sum(len(images[g][b]) for g in images.keys() for b in BANDS)
        print(f"\nTotal images loaded: {total}")
        # Save original brightest/darkest plot
        plt.figure()
        plot_original_brightest_el(images)
        plt.savefig(grid_dir / 'el_brightest_darkest_original.png', dpi=150, bbox_inches='tight')
        plt.close()
        # Normalize images
        print("\n[2/4] Normalizing images by group and band...")
        normalized_images = normalize_images(images)
        print("="*80)
        print(f"IMAGE NORMALIZATION BY GROUP AND BAND for {image_folder}")
        print("="*80)
        # Display comparison
        print("\n[3/4] Creating visualization...")
        plt.figure()
        display_comparison(normalized_images)
        plt.savefig(grid_dir / 'normalization_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        plt.figure()
        plot_normalized_brightest_el(normalized_images)
        plt.savefig(grid_dir / 'el_brightest_darkest.png', dpi=150, bbox_inches='tight')
        plt.close()
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
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
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
                plt.savefig(grid_dir / f'{group}_{band}_grid.jpg', dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved: {grid_dir}/{group}_{band}_grid.jpg')
if __name__ == '__main__':
    main()
