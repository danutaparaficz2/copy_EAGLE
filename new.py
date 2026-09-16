# import numpy as np
# from PIL import Image

# img = Image.open("/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/24-P10-A/EL/24-P10_A1_EL_Cell011_normalized.tif")
# arr = np.array(img)

# print(f"PIL Mode: {img.mode}")
# print(f"Data type: {arr.dtype}")
# print(f"Shape: {arr.shape}")
# print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.2f}")

import os
from PIL import Image
import io

def get_b64_kb(path):
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return len(buf.getvalue()) * 4 / 3 / 1024  # Base64 size in KB

vi_kb = get_b64_kb("/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/24-P10-A/VI/24-P10_A1_VI_Cell014_normalized.tif")
el_kb = get_b64_kb("/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/24-P10-A/EL/24-P10_A1_EL_Cell014_normalized.tif")

print(f"VI Payload: {vi_kb:.1f} KB  |  EL Payload: {el_kb:.1f} KB")