#!/usr/bin/env python3
"""
Merge two classification CSVs (EL and VI) by cell ID.

Assumes filenames like:
  23-P09-D1_EL_Cell001_normalized.tif
  23-P09-D1_VI_Cell001_normalized.tif

Two modes:
1) CSV (default): simplified rows `key,<labels>`
   - <labels>: follows rules defined in `simplify_labels()`
2) Arrays mode: load EL and VI (optional) images as NumPy arrays and save per-key `.npz`
    - EL converted to grayscale (H,W)
    - VI converted to RGB (H,W,3) if available; otherwise uses EL only
    - Stack channel-first: [EL(1), VI(3)] -> shape (4,H,W) or [EL(1)] -> shape (1,H,W)
    - Requires flags: `--el-dir`, `--vi-dir`, `--arrays-out <dir>`
3) Arrays-to-Pickle mode: write a pickle containing a list of dict rows
    - Stored as Python list: [{'image': np.ndarray (4,244,244) or (1,244,244), 'label': int|list[int]}, ...]
    - Label is int or list[int] depending on simplified rules
    - Requires flags: `--el-dir`, `--vi-dir`, `--arrays-pkl-out <file>`

- key: sample+cell identifier (e.g., 23-P09-D1_Cell001)
- el_classification: integer code(s) from EL CSV (may be comma-separated)
- vi_classification: integer code from VI CSV

Usage:
  python merge_classification_csvs.py \
    classification_results_23_P09-D_integer.csv \
    classification_results_VI_23_P09-D.csv \
    classification_results_EL_VI_23_P09-D_merged.csv
"""

import csv
import re
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, List
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import pickle
import math
from collections import defaultdict
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False

# Target spatial size for all modality images (width, height)
TARGET_SIZE = (224, 224)  # (W, H)

FILENAME_RE = re.compile(r"^(?P<sample>.+?)_(?P<modality>EL|VI)_(?P<cell>Cell\d+)(?:_normalized)?\.tif$")


def parse_filename(fname: str) -> Optional[Tuple[str, str, str]]:
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    sample = m.group('sample')
    modality = m.group('modality')
    cell = m.group('cell')
    return sample, modality, cell


def read_csv_to_map(path: Path) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Returns mapping: key -> (sample, modality, cell, classification)
    where key = f"{sample}_{cell}" (shared by EL and VI for the same cell)
    """
    mapping: Dict[str, Tuple[str, str, str, str]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('filename') or row.get('file') or ''
            cls = row.get('classification', '')
            parsed = parse_filename(fname)
            if not parsed:
                # Skip rows that don't match expected filename pattern
                continue
            sample, modality, cell = parsed
            key = f"{sample}_{cell}"
            mapping[key] = (sample, modality, cell, cls)
    return mapping


def select_combined_label(el_class: str, vi_class: str) -> str:
    """Return combined label using rule: use VI unless VI is 0/empty then EL.
    If EL has multiple codes (e.g. "1,3") and VI is zero, keep them all.
    If both empty, return empty string."""
    vc = (vi_class or '').strip()
    if vc and vc not in ('0', '0,'):  # treat any non-zero as preferred
        return vc
    ec = (el_class or '').strip()
    return ec


def simplify_labels(el_class: str, vi_class: str) -> str:
    """Return simplified label string following user rule.
    Cases:
            EL=0 VI=0           -> 0
            EL=0 VI=5           -> 5 (VI preferred; drop the zero)
            EL=3 VI=0           -> 3
            EL=1,3 VI=0         -> 1,3
            EL=1 VI=5           -> 1,5
            EL="" VI=5          -> 5
    If EL equals VI (string match) output single value.
        If VI empty or '0' and EL non-empty output EL.
        If EL empty or EL=='0' and VI non-zero output VI.
        Else output EL + ',' + VI.
    """
    el = (el_class or '').strip()
    vi = (vi_class or '').strip()
    if not el and not vi:
        return ''
    # Treat zero equivalently across multi formatting
    def is_zero(val: str) -> bool:
        return val == '0' or val == '0,' or val == '0,0'
    if el == vi:
        return el
    if (is_zero(vi) or not vi) and el:
        return el
    if (not el or is_zero(el)) and vi and not is_zero(vi):
        # Prefer VI alone when EL is missing or zero
        return vi
    # Both present and different
    return f"{el},{vi}".rstrip(',')


def merge_csvs(el_csv: str, vi_csv: str, out_csv: str) -> None:
    el_path = Path(el_csv)
    vi_path = Path(vi_csv)
    out_path = Path(out_csv)

    if not el_path.exists():
        raise FileNotFoundError(f"EL CSV not found: {el_csv}")
    if not vi_path.exists():
        raise FileNotFoundError(f"VI CSV not found: {vi_csv}")

    el_map = read_csv_to_map(el_path)
    vi_map = read_csv_to_map(vi_path)

    # Build union of keys
    all_keys = sorted(set(el_map.keys()) | set(vi_map.keys()), key=lambda k: (k.split('_')[-1].lower(), k.lower()))

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for key in all_keys:
            # sample and cell from either side
            sample = None
            cell = None

            el_fname = ''
            el_class = ''
            if key in el_map:
                s, modality, c, cls = el_map[key]
                sample = s
                cell = c
                el_fname = f"{s}_EL_{c}_normalized.tif"
                el_class = cls

            vi_fname = ''
            vi_class = ''
            if key in vi_map:
                s, modality, c, cls = vi_map[key]
                sample = sample or s
                cell = cell or c
                vi_fname = f"{s}_VI_{c}_normalized.tif"
                vi_class = cls

            simplified = simplify_labels(el_class, vi_class)
            writer.writerow([key, simplified])

    print(f"✓ Merged {len(all_keys)} rows")
    print(f"✓ Output: {out_csv}")


def build_modality_filename(sample: str, modality: str, cell: str) -> str:
    return f"{sample}_{modality}_{cell}_normalized.tif"


def read_el_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = img.convert('L')
    return np.array(img)


def read_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = img.convert('RGB')
    return np.array(img)


def export_arrays_for_keys(keys, el_dir: Path, vi_dir: Path, out_dir: Path, no_vi: bool = False, no_el: bool = False) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    iterator = tqdm(keys, total=len(keys), desc="Export .npz") if _HAS_TQDM else keys
    for key in iterator:
        # key format: "{sample}_{cell}". Split on last underscore to be safe.
        us = key.rfind('_')
        if us == -1:
            skipped += 1
            continue
        sample = key[:us]
        cell = key[us+1:]

        def find_modality_file(base_dir: Path, modality: str) -> Path:
            # Try common filename patterns first
            candidates = [
                f"{sample}_{modality}_{cell}_normalized.tif",
                f"{sample}_{modality}_{cell}.tif",
                f"{sample}_{modality}_{cell}.tiff",
                f"{sample}_{modality}_{cell}_normalized.png",
                f"{sample}_{modality}_{cell}.png",
                f"{sample}_{modality}_{cell}_normalized.jpg",
                f"{sample}_{modality}_{cell}.jpg",
            ]
            for name in candidates:
                p = base_dir / name
                if p.exists():
                    return p
            # Glob fallback
            for pat in [
                f"{sample}_{modality}_{cell}_normalized.*",
                f"{sample}_{modality}_{cell}.*",
            ]:
                m = list(base_dir.glob(pat))
                if m:
                    return m[0]
            raise FileNotFoundError(base_dir / candidates[0])

        try:
            TW, TH = TARGET_SIZE
            
            if no_el:
                # VI only mode: load VI as RGB (3,H,W)
                vi_path = find_modality_file(vi_dir, 'VI')
                vi_arr = read_rgb(vi_path)
                if vi_arr.shape[:2] != (TH, TW):
                    vi_arr = np.array(Image.fromarray(vi_arr).resize((TW, TH), Image.BILINEAR))
                # VI only: 3xHxW
                img_combined = np.transpose(vi_arr, (2, 0, 1))
                el_arr = None
            else:
                # EL required (with optional VI)
                el_path = find_modality_file(el_dir, 'EL')
                el_arr = read_el_gray(el_path)          # (H, W) uint8
                
                # Try to find VI; if missing or --no-vi flag set, proceed with EL only
                vi_arr = None
                if not no_vi:
                    try:
                        vi_path = find_modality_file(vi_dir, 'VI')
                        vi_arr = read_rgb(vi_path)              # (H, W, 3) uint8
                    except FileNotFoundError:
                        pass  # VI missing, will use EL only

                # Resize all modalities to TARGET_SIZE (W,H)
                if el_arr.shape[:2] != (TH, TW):
                    el_arr = np.array(Image.fromarray(el_arr).resize((TW, TH), Image.BILINEAR))
                
                if vi_arr is not None:
                    if vi_arr.shape[:2] != (TH, TW):
                        vi_arr = np.array(Image.fromarray(vi_arr).resize((TW, TH), Image.BILINEAR))
                    # Stack into channel-first 4xHxW: [EL(1), VI(3)]
                    el_ch = el_arr[np.newaxis, :, :]
                    vi_ch = np.transpose(vi_arr, (2, 0, 1))
                    img_combined = np.concatenate([el_ch, vi_ch], axis=0)
                else:
                    # EL only: 1xHxW
                    img_combined = el_arr[np.newaxis, :, :]
        except FileNotFoundError as e:
            print(f"! Missing file for {key}: {e}")
            skipped += 1
            continue
        except Exception as e:
            print(f"! Error reading images for {key}: {e}")
            skipped += 1
            continue

        npz_path = out_dir / f"{key}.npz"
        try:
            # Save combined array (4, 3, or 1, H, W). Keep individual arrays for convenience.
            if no_el:
                np.savez_compressed(npz_path, img=img_combined, vi=vi_arr)
            elif vi_arr is not None:
                np.savez_compressed(npz_path, img=img_combined, el=el_arr, vi=vi_arr)
            else:
                np.savez_compressed(npz_path, img=img_combined, el=el_arr)
            written += 1
            if not _HAS_TQDM:
                print(f"✓ Saved {npz_path.name}")
        except Exception as e:
            print(f"! Error saving {npz_path.name}: {e}")
            skipped += 1
    return written, skipped


def create_grid_visualization(images: List[str], category: str, output_path: str, thumb_size: int = 360, cols: int = 5):
    """Create a grid visualization of images."""
    if not images:
        print(f"  No images for category: {category}")
        return
    
    n = len(images)
    rows = math.ceil(n / cols)
    
    # Create canvas
    pad = 24
    title_pad = 80
    canvas_width = cols * thumb_size + (cols + 1) * pad
    canvas_height = rows * thumb_size + (rows + 1) * pad + title_pad
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    # Draw title
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
    except:
        font = ImageFont.load_default()
    
    title = f"{category} ({n} images)"
    draw.text((pad, 10), title, fill="black", font=font)
    
    # Place thumbnails
    for idx, img_path in enumerate(images):
        try:
            img = Image.open(img_path)
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            
            row = idx // cols
            col = idx % cols
            x = col * thumb_size + (col + 1) * pad
            y = row * thumb_size + (row + 1) * pad + title_pad
            
            canvas.paste(img, (x, y))
        except Exception as e:
            print(f"    Error loading {img_path}: {e}")
    
    canvas.save(output_path, quality=95)
    print(f"  ✓ Saved visualization: {output_path}")


def arrays_to_pickle_for_keys(
    keys,
    el_dir: Path,
    vi_dir: Path,
    out_pkl: Path,
    labels_by_key: Dict[str, str],
    no_vi: bool = False,
    no_el: bool = False,
) -> Tuple[int, int]:
    """Produce list of {'image': <np.ndarray 4x244x244>, 'label': int|list[int]} and pickle dump it.
    Also creates a CSV file with filename and label.
    Returns (written_items, skipped_keys)."""
    dataset = []
    written = 0
    skipped = 0
    iterator = tqdm(keys, total=len(keys), desc="Arrays PKL") if _HAS_TQDM else keys
    for key in iterator:
        us = key.rfind('_')
        if us == -1:
            skipped += 1
            continue
        sample = key[:us]
        cell = key[us+1:]

        def find_modality_file(base_dir: Path, modality: str) -> Path:
            candidates = [
                f"{sample}_{modality}_{cell}_normalized.tif",
                f"{sample}_{modality}_{cell}.tif",
                f"{sample}_{modality}_{cell}.tiff",
                f"{sample}_{modality}_{cell}_normalized.png",
                f"{sample}_{modality}_{cell}.png",
                f"{sample}_{modality}_{cell}_normalized.jpg",
                f"{sample}_{modality}_{cell}.jpg",
            ]
            for name in candidates:
                p = base_dir / name
                if p.exists():
                    return p
            for pat in [
                f"{sample}_{modality}_{cell}_normalized.*",
                f"{sample}_{modality}_{cell}.*",
            ]:
                m = list(base_dir.glob(pat))
                if m:
                    return m[0]
            raise FileNotFoundError(base_dir / candidates[0])

        try:
            TW, TH = TARGET_SIZE
            
            if no_el:
                # VI only mode: load VI as RGB (3,H,W)
                vi_path = find_modality_file(vi_dir, 'VI')
                vi_arr = read_rgb(vi_path)
                if vi_arr.shape[:2] != (TH, TW):
                    vi_arr = np.array(Image.fromarray(vi_arr).resize((TW, TH), Image.BILINEAR))
                # VI only: 3xHxW
                img_combined = np.transpose(vi_arr, (2, 0, 1))
            else:
                # EL required (with optional VI)
                el_path = find_modality_file(el_dir, 'EL')
                el_arr = read_el_gray(el_path)
                
                # Try to find VI; if missing or --no-vi flag set, proceed with EL only
                vi_arr = None
                if not no_vi:
                    try:
                        vi_path = find_modality_file(vi_dir, 'VI')
                        vi_arr = read_rgb(vi_path)
                    except FileNotFoundError:
                        pass  # VI missing, will use EL only

                if el_arr.shape[:2] != (TH, TW):
                    el_arr = np.array(Image.fromarray(el_arr).resize((TW, TH), Image.BILINEAR))
                
                if vi_arr is not None:
                    if vi_arr.shape[:2] != (TH, TW):
                        vi_arr = np.array(Image.fromarray(vi_arr).resize((TW, TH), Image.BILINEAR))
                    # Stack into channel-first 4xHxW: [EL(1), VI(3)]
                    el_ch = el_arr[np.newaxis, :, :]
                    vi_ch = np.transpose(vi_arr, (2, 0, 1))
                    img_combined = np.concatenate([el_ch, vi_ch], axis=0)
                else:
                    # EL only: 1xHxW
                    img_combined = el_arr[np.newaxis, :, :]

            label_str = (labels_by_key.get(key) or '').strip()
            label_vals = [int(p) for p in label_str.split(',') if p.strip() != ''] if label_str else []
            label_obj = label_vals[0] if len(label_vals) == 1 else label_vals

            dataset.append([img_combined, label_obj])
            written += 1
            if not _HAS_TQDM:
                print(f"✓ Packed {key}")
        except FileNotFoundError as e:
            print(f"! Missing file for {key}: {e}")
            skipped += 1
            continue
        except Exception as e:
            print(f"! Error processing {key}: {e}")
            skipped += 1
            continue

    try:
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        import sys
        # Numpy compatibility: handle both old and new numpy versions
        if hasattr(np, '_core'):
            # numpy 2.x: map old names to new
            sys.modules['numpy.core'] = sys.modules.get('numpy._core', np._core)
            sys.modules['numpy.core.numeric'] = sys.modules.get('numpy._core.numeric', np._core.numeric)
        else:
            # numpy 1.x: map new names to old
            sys.modules['numpy._core'] = sys.modules.get('numpy.core', np.core)
            sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric', np.core.numeric)
        with open(out_pkl, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Pickle list saved: {out_pkl} (items={written}, skipped={skipped})")
        
        # Also create CSV with filename and label
        csv_path = out_pkl.with_suffix('.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])  # Header
            for key in keys:
                label_str = labels_by_key.get(key, '')
                if label_str:  # Only write if we have a label
                    writer.writerow([key, label_str])
        print(f"✓ CSV saved: {csv_path}")
        
        # Create grid visualizations for each class and multi-label combinations
        # Map integer labels to category names
        label_names = {
            0: "Good",
            1: "Crack",
            2: "Cross",
            3: "Dark",
            4: "Corrosion",
            5: "Discoloration",
            6: "Delamination"
        }
        
        # Group images by exact label combination
        panel_dir = out_pkl.parent
        vis_dir = panel_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Collect image paths per category (exact label combination)
        category_images = defaultdict(list)
        
        for key in keys:
            label_str = labels_by_key.get(key, '').strip()
            if not label_str:
                continue
            
            # Parse label values
            label_vals = sorted([int(p) for p in label_str.split(',') if p.strip() != '']) if label_str else []
            if not label_vals:
                continue
            
            us = key.rfind('_')
            if us == -1:
                continue
            sample = key[:us]
            cell = key[us+1:]
            
            # Create category name based on label combination
            if len(label_vals) == 1:
                # Single label
                category = label_names.get(label_vals[0], f"Unknown_{label_vals[0]}")
            else:
                # Multi-label: join names with underscore
                category = "_and_".join([label_names.get(lv, f"Unknown_{lv}") for lv in label_vals])
            
            try:
                # Try to add EL image
                el_path = find_modality_file(el_dir, 'EL')
                if el_path.exists():
                    category_images[category].append(str(el_path))
                
                # Try to add VI image if available
                if not no_vi:
                    try:
                        vi_path = find_modality_file(vi_dir, 'VI')
                        if vi_path.exists():
                            category_images[category].append(str(vi_path))
                    except FileNotFoundError:
                        pass  # VI not available, skip
            except Exception as e:
                print(f"! Error processing images for {key}: {e}")
                continue
        
        # Create grid visualizations
        print(f"\n✓ Creating visualizations in {vis_dir}/")
        for category, images in sorted(category_images.items()):
            if images:
                output_path = vis_dir / f"{category}.png"
                create_grid_visualization(images, category, str(output_path))
                print(f"  {category}: {len(images)} images")
        
    except Exception as e:
        print(f"! Error saving files: {e}")
    return written, skipped


if __name__ == '__main__':
    # Defaults
    default_panel = '23-P09-D'

    # Parse args with simple flag handling
    argv = sys.argv[1:]
    no_vi = False
    no_el = False
    panel = None

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--no-vi':
            no_vi = True
            i += 1
            continue
        if a == '--no-el':
            no_el = True
            i += 1
            continue
        # First positional argument is panel
        if panel is None:
            panel = a
        i += 1

    # Use default panel if not provided
    if panel is None:
        panel = default_panel
    
    # Infer paths from panel name
    el_dir = Path(f'/Users/eagle/Documents/eagle-classification/normalized_images/{panel}/EL')
    vi_dir = Path(f'/Users/eagle/Documents/eagle-classification/normalized_images/{panel}/VI')
    arrays_out = Path(f'/Users/eagle/Documents/eagle-classification/OPENAI/{panel}')
    arrays_pkl_out = Path(f'/Users/eagle/Documents/eagle-classification/OPENAI/{panel}/dataset_arrays_{panel}.pkl')
    
    el_csv = f'/Users/eagle/Documents/eagle-classification/OPENAI/{panel}/classification_results_EL_{panel}_integer.csv'
    vi_csv = f'/Users/eagle/Documents/eagle-classification/OPENAI/{panel}/classification_results_VI_{panel}_integer.csv'
    out_csv = f'/Users/eagle/Documents/eagle-classification/OPENAI/{panel}/classification_results_EL_VI_{panel}_simple.csv'

    try:
        # Build keys from CSVs first (union)
        el_map = read_csv_to_map(Path(el_csv))
        vi_map = read_csv_to_map(Path(vi_csv)) if Path(vi_csv).exists() else {}
        all_keys = sorted(set(el_map.keys()) | set(vi_map.keys()), key=lambda k: (k.split('_')[-1].lower(), k.lower()))
        
        # Filter out D5 group completely
        keys = [k for k in all_keys if 'D5' not in k and '-D5_' not in k]
        print(f"✓ Filtered out D5 group: {len(all_keys)} -> {len(keys)} keys")

        # Priority: pickle > npz when respective flags present
        if el_dir and vi_dir and arrays_pkl_out:
            # Add _no_vi or _no_el suffix to output filename if flag is set
            if no_vi:
                pkl_path = arrays_pkl_out.parent / f"{arrays_pkl_out.stem}_no_vi{arrays_pkl_out.suffix}"
            elif no_el:
                pkl_path = arrays_pkl_out.parent / f"{arrays_pkl_out.stem}_no_el{arrays_pkl_out.suffix}"
            else:
                pkl_path = arrays_pkl_out
            
            labels_by_key: Dict[str, str] = {}
            for key in keys:
                el_class = el_map.get(key, ('','','',''))[3] if key in el_map else ''
                vi_class = vi_map.get(key, ('','','',''))[3] if key in vi_map else ''
                # If --no-vi flag is set, only use EL labels
                # If --no-el flag is set, only use VI labels
                if no_vi:
                    labels_by_key[key] = el_class
                elif no_el:
                    labels_by_key[key] = vi_class
                else:
                    labels_by_key[key] = simplify_labels(el_class, vi_class)
            written, skipped = arrays_to_pickle_for_keys(keys, el_dir, vi_dir, pkl_path, labels_by_key, no_vi, no_el)
            print(f"✓ Array-PKL items: {written}, skipped: {skipped}")
            print(f"✓ Output file: {pkl_path}")
        # Else if arrays mode requested and dirs provided, export .npz arrays per key
        elif el_dir and vi_dir and arrays_out:
            # Add _no_vi or _no_el suffix to output directory if flag is set
            if no_vi:
                npz_dir = arrays_out.parent / f"{arrays_out.name}_no_vi"
            elif no_el:
                npz_dir = arrays_out.parent / f"{arrays_out.name}_no_el"
            else:
                npz_dir = arrays_out
            
            written, skipped = export_arrays_for_keys(keys, el_dir, vi_dir, npz_dir, no_vi, no_el)
            print(f"✓ Arrays saved: {written}, skipped: {skipped}")
            print(f"✓ Output dir: {npz_dir}")
        else:
            # Fall back to simplified CSV using earlier pipeline
            # Reconstruct a temp file using simplified rows
            with open(out_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                iterator = tqdm(keys, total=len(keys), desc="Merge CSV") if _HAS_TQDM else keys
                for key in iterator:
                    el_class = el_map.get(key, ('','','',''))[3] if key in el_map else ''
                    vi_class = vi_map.get(key, ('','','',''))[3] if key in vi_map else ''
                    simplified = simplify_labels(el_class, vi_class)
                    writer.writerow([key, simplified])
                    if not _HAS_TQDM:
                        print(f"✓ Wrote {key}")
            print(f"✓ Merged {len(keys)} rows")
            print(f"✓ Output: {out_csv}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
