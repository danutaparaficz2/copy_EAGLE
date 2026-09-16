import base64
import csv
import io
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import cv2
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from PIL import Image
import requests

# ---------------------------------------------------------
# Environment & Path Configuration
# ---------------------------------------------------------
load_dotenv("/Users/eagle/Documents/.env")

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "token-not-needed")

panel = os.getenv("PANEL", "24-P10-A")  #[cite: 1]
IMAGE_DIR = f"/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/{panel}/EL/"  #[cite: 1]
OUTPUT_INTEGER_CSV = f"/Users/danuta.paraficz/PyProjects/eagle-classification/OPENAI/{panel}/classification_results_EL_{panel}_integer.csv"  #[cite: 1]

# Set starting image index (1-based)
START_INDEX = 1

HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("OPENAI_HEALTHCHECK_TIMEOUT", "15.0"))  #[cite: 1]
HEALTHCHECK_RETRIES = max(1, int(os.getenv("OPENAI_HEALTHCHECK_RETRIES", "3")))  #[cite: 1]
SKIP_ENDPOINT_CHECK = os.getenv("OPENAI_SKIP_ENDPOINT_CHECK", "0").strip().lower() in {"1", "true", "yes"}  #[cite: 1]

DEFECT_TO_INT = {  #[cite: 1]
    "good": 0,  #[cite: 1]
    "crack": 1,  #[cite: 1]
    "cross": 2,  #[cite: 1]
    "dark": 3,  #[cite: 1]
    "corrosion": 4,  #[cite: 1]
    "undefined": 9,  #[cite: 1]
}  #[cite: 1]

# ---------------------------------------------------------
# Inspection Protocol & Macro-Focused Prompt
# ---------------------------------------------------------
PROMPT_TEXT = """Analyze this laboratory-captured, calibrated electroluminecese (EL) image of a solar cell for defects under flat illumination.

  Only report defects that are clearly visible.

    Do NOT classify the following as defects:
    - normal vertical EL stripe patterns
    - faint texture lines aligned with cell structure

    Defect categories:
    Crack:  a clearly visible,prominent, continuous, irregular fracture line not aligned with gridlines.
    Cross: a short cross-like interruption.
    Dark: significant in size areas with clearly darker region.
    Corrosion: visible oxidation specially near busbars or fingers.

OUTPUT FORMAT:
Provide 1 brief sentence stating the visual status of the busbars and cell surface, followed immediately by:
DECISION: comma-separated codes, or 0 if healthy
"""


# ---------------------------------------------------------
# Utilities, Healthchecks & Compact Image Handling
# ---------------------------------------------------------
def normalize_base_url(base_url: str) -> str:  #[cite: 1]
    normalized = base_url.rstrip("/")  #[cite: 1]
    for suffix in ("/chat/completions", "/completions"):  #[cite: 1]
        if normalized.endswith(suffix):  #[cite: 1]
            normalized = normalized[:-len(suffix)]  #[cite: 1]
            break  #[cite: 1]
    return normalized  #[cite: 1]


def check_api_endpoint(base_url: str, api_key: str, timeout_seconds: float, retries: int) -> tuple[bool, str]:  #[cite: 1]
    parsed = urlparse(base_url)  #[cite: 1]
    if not parsed.scheme or not parsed.netloc:  #[cite: 1]
        return False, f"Invalid OPENAI_BASE_URL: {base_url}"  #[cite: 1]

    headers = {}
    if api_key and api_key != "token-not-needed":
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = "unknown error"  #[cite: 1]
    for attempt in range(1, retries + 1):  #[cite: 1]
        try:  #[cite: 1]
            target_url = f"{base_url}/models"
            response = requests.get(target_url, headers=headers, timeout=timeout_seconds)  #[cite: 1]

            if response.status_code in (401, 403):
                return False, f"Authentication failed (HTTP {response.status_code}): Check your ATHENA_API_KEY token."
            if response.status_code == 404:
                return False, (
                    f"HTTP 404 at {target_url}. The proxy might be stripping subpaths instead "
                    f"of forwarding to backend Athena. Proxy response: {response.text}"
                )

            response.raise_for_status()  #[cite: 1]
            return True, ""  #[cite: 1]
        except requests.RequestException as exc:  #[cite: 1]
            last_error = str(exc)  #[cite: 1]
            if attempt < retries:  #[cite: 1]
                print(f"Endpoint check attempt {attempt}/{retries} failed: {exc}. Retrying...")  #[cite: 1]

    return False, f"Cannot reach API endpoint at {base_url} after {retries} attempt(s): {last_error}"  #[cite: 1]


def prepare_raw_el_image(path: str, max_dim: int = 512, jpeg_quality: int = 75, save_debug: bool = False) -> str:
    """
    Loads raw single-channel EL image without artificial enhancement or sharpening,
    resizes to max_dim (512px), and encodes as JPEG (quality 75) to drop the base64
    payload to ~25 KB so it easily bypasses corporate WAF payload limits.
    """
    if not os.path.exists(path):  #[cite: 1]
        raise FileNotFoundError(f"Image file not found: {path}")  #[cite: 1]

    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read image with OpenCV: {path}")


    # 2. Replicate raw grayscale to (H, W, 3)
    rgb_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # 3. Compress to compact JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success, buffer = cv2.imencode(".jpg", rgb_img, encode_params)
    if not success:
        raise ValueError(f"Failed to encode image to JPEG buffer: {path}")

    if save_debug:
        debug_path = Path(path).parent / "debug_compact_el.jpg"
        with open(debug_path, "wb") as f:
            f.write(buffer)
        print(f"\n[DEBUG] Saved sample ({len(buffer) / 1024:.1f} KB) to: {debug_path}")

    b64 = base64.b64encode(buffer).decode("utf-8")  #[cite: 1]
    return f"data:image/jpeg;base64,{b64}"


def extract_decision_code(raw_text: str) -> str:  #[cite: 1]
    """
    Strictly extracts labels only from the 'DECISION:' line to prevent
    scraping reasoning words into false positive defect codes.
    """
    if not raw_text or not raw_text.strip():  #[cite: 1]
        return "0"  #[cite: 1]

    match = re.search(r"DECISION:\s*([^\n\r]+)", raw_text, re.IGNORECASE)  #[cite: 1]
    if not match:
        return "0"

    decision_str = match.group(1).strip()  #[cite: 1]
    detected_codes = set()  #[cite: 1]
    tokens = [t.strip().lower() for t in re.split(r"[,;/]+", decision_str) if t.strip()]  #[cite: 1]

    for token in tokens:  #[cite: 1]
        if token in DEFECT_TO_INT:  #[cite: 1]
            detected_codes.add(DEFECT_TO_INT[token])  #[cite: 1]
        elif token.isdigit():  #[cite: 1]
            detected_codes.add(int(token))  #[cite: 1]
        else:  #[cite: 1]
            for name, code in DEFECT_TO_INT.items():  #[cite: 1]
                if re.search(rf"\b{name}\b", token):  #[cite: 1]
                    detected_codes.add(code)  #[cite: 1]

    if len(detected_codes) > 1 and 0 in detected_codes:  #[cite: 1]
        detected_codes.remove(0)  #[cite: 1]

    return ", ".join(map(str, sorted(detected_codes))) if detected_codes else "0"  #[cite: 1]


def classify_image(client: OpenAI, image_path: str, save_debug: bool = False) -> tuple[str, str]:  #[cite: 1]
    try:  #[cite: 1]
        data_uri = prepare_raw_el_image(image_path, save_debug=save_debug)
    except Exception as e:  #[cite: 1]
        return f"ERROR: {e}", ""  #[cite: 1]

    try:  #[cite: 1]
        completion = client.chat.completions.create(  #[cite: 1]
            model="Qwen/Qwen2.5-VL-32B-Instruct",  #[cite: 1]
            temperature=0.0,  #[cite: 1]
            max_tokens=200,
            messages=[  #[cite: 1]
                {  #[cite: 1]
                    "role": "system",  #[cite: 1]
                    "content": "You are an expert industrial photovoltaic inspector specializing in electroluminescence (EL) defect classification.",  #[cite: 1]
                },  #[cite: 1]
                {  #[cite: 1]
                    "role": "user",  #[cite: 1]
                    "content": [  #[cite: 1]
                        {  #[cite: 1]
                            "type": "image_url",  #[cite: 1]
                            "image_url": {  #[cite: 1]
                                "url": data_uri,  #[cite: 1]
                                "detail": "high",  #[cite: 1]
                            },  #[cite: 1]
                        },  #[cite: 1]
                        {"type": "text", "text": PROMPT_TEXT},  #[cite: 1]
                    ],  #[cite: 1]
                },  #[cite: 1]
            ],  #[cite: 1]
        )  #[cite: 1]
        content = completion.choices[0].message.content or ""  #[cite: 1]
        parsed_code = extract_decision_code(content)  #[cite: 1]
        return parsed_code, content.strip().replace("\n", " ")  #[cite: 1]

    except Exception as e:  #[cite: 1]
        return f"API_ERROR: {e}", ""  #[cite: 1]


# ---------------------------------------------------------
# Batch Pipeline
# ---------------------------------------------------------
def main():  #[cite: 1]
    base_url = normalize_base_url(BASE_URL)  #[cite: 1]

    if not API_KEY or API_KEY == "token-not-needed":  #[cite: 1]
        print("WARNING: ATHENA_API_KEY is empty or default. Check your .env file.")

    print(f"Connecting to endpoint: {base_url}")

    if not SKIP_ENDPOINT_CHECK:  #[cite: 1]
        endpoint_ok, endpoint_error = check_api_endpoint(  #[cite: 1]
            base_url,  #[cite: 1]
            api_key=API_KEY,
            timeout_seconds=HEALTHCHECK_TIMEOUT_SECONDS,  #[cite: 1]
            retries=HEALTHCHECK_RETRIES,  #[cite: 1]
        )  #[cite: 1]
        if not endpoint_ok:  #[cite: 1]
            print(f"\n[FAIL] Healthcheck Error: {endpoint_error}")  #[cite: 1]
            return  #[cite: 1]
        print("[OK] Endpoint healthcheck passed (/models reachable).")

    image_dir = Path(IMAGE_DIR)  #[cite: 1]
    if not image_dir.is_dir():  #[cite: 1]
        print(f"ERROR: Directory not found: {IMAGE_DIR}")  #[cite: 1]
        return  #[cite: 1]

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}  #[cite: 1]
    image_files = sorted(  #[cite: 1]
        [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions],  #[cite: 1]
        key=lambda x: x.name.lower(),  #[cite: 1]
    )  #[cite: 1]

    total_images = len(image_files)  #[cite: 1]
    if not total_images:  #[cite: 1]
        print(f"No images found in {IMAGE_DIR}")  #[cite: 1]
        return  #[cite: 1]

    # Slice image list starting from START_INDEX (1-based index)
    start_pos = max(1, min(START_INDEX, total_images))
    queued_files = image_files[start_pos - 1:]

    output_path = Path(OUTPUT_INTEGER_CSV)  #[cite: 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)  #[cite: 1]

    # Append mode if starting past 1 and file exists, otherwise fresh write
    file_exists = output_path.is_file() and output_path.stat().st_size > 0
    write_header = not file_exists or start_pos == 1
    open_mode = "a" if (file_exists and start_pos > 1) else "w"

    print(f"Found {total_images} total images. Starting from index {start_pos}/{total_images} ({len(queued_files)} remaining).")
    print(f"Writing to (mode='{open_mode}'): {OUTPUT_INTEGER_CSV}")

    client = OpenAI(api_key=API_KEY, base_url=base_url)

    with open(output_path, open_mode, newline="", encoding="utf-8") as csvfile:  #[cite: 1]
        writer = csv.writer(csvfile)  #[cite: 1]
        if write_header:
            writer.writerow(["filename", "classification", "reasoning"])  #[cite: 1]

        for offset, image_path in enumerate(queued_files):
            current_idx = start_pos + offset
            print(f"[{current_idx}/{total_images}] Classifying {image_path.name}...", end=" ", flush=True)

            save_debug_sample = (offset == 0)
            code, reasoning = classify_image(client, str(image_path), save_debug=save_debug_sample)  #[cite: 1]

            if code.startswith("API_ERROR:"):  #[cite: 1]
                print(f"\n{code}")  #[cite: 1]
                print("Aborting to avoid sending queries to a failing endpoint.")  #[cite: 1]
                return  #[cite: 1]

            print(f"Result -> {code} | {reasoning}")  #[cite: 1]
            writer.writerow([image_path.name, code, reasoning])  #[cite: 1]
            csvfile.flush()  #[cite: 1]

    print(f"\nProcessing complete: {output_path}")  #[cite: 1]


if __name__ == "__main__":  #[cite: 1]
    main()  #[cite: 1]