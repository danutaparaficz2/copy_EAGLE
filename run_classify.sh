#!/bin/bash
set -e

#for panel in C14-A C14-B C14-C C14-D C14-E C14-F C14-G C14-H; do
for panel in 23-P09H 24-128-A; do
    echo "Processing panel: $panel"
    python classify_VI.py --panel "$panel"
    #python classify_VIT.py --panel "$panel"
done

echo "All panels processed."
