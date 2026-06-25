#!/bin/bash
set -e

for panel in C14-B C14-C C14-D C14-E C14-F C14-G C14-H; do
    echo "Processing panel: $panel"
    python load_and_normalize_images.py --panel "$panel"
done

echo "All panels processed."
