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
# Load custom .env path; falls back gracefully if not found
load_dotenv('/Users/eagle/Documents/.env')

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "token-not-needed")
print(BASE_URL)
if not API_KEY:
    raise ValueError("ATHENA_API_KEY is missing! Check your .env file.")
print(f"Loaded BASE_URL: {BASE_URL}")
print(f"Loaded API_KEY:  {API_KEY[:10]}... (Total length: {len(API_KEY)})")
panel = os.getenv("PANEL", "23-P09-D")
IMAGE_DIR = f"/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/{panel}/VI/"
OUTPUT_INTEGER_CSV = f"/Users/danuta.paraficz/PyProjects/eagle-classification/OPENAI/{panel}/classification_results_VI_{panel}_integer.csv"

HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("OPENAI_HEALTHCHECK_TIMEOUT", "15.0"))
HEALTHCHECK_RETRIES = max(1, int(os.getenv("OPENAI_HEALTHCHECK_RETRIES", "3")))
SKIP_ENDPOINT_CHECK = os.getenv("OPENAI_SKIP_ENDPOINT_CHECK", "0").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------
# Utilities & Healthchecks
# ---------------------------------------------------------
def normalize_base_url(base_url: str) -> str:
    """Strips trailing slashes and specific subpath suffixes if accidentally included."""
    normalized = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    return normalized

def check_api_endpoint(base_url: str, timeout_seconds: float, retries: int) -> tuple[bool, str]:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return False, f"Invalid OPENAI_BASE_URL: {base_url}"

    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{base_url}/models", timeout=timeout_seconds)
            response.raise_for_status()
            return True, ""
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < retries:
                print(f"Endpoint check attempt {attempt}/{retries} failed: {exc}. Retrying...")

    return False, f"Cannot reach API endpoint at {base_url} after {retries} attempt(s): {last_error}"


def encode_file_to_data_uri(path: str) -> str:
    """Loads image, converts mode if needed, and encodes as a base64 PNG data URI."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_decision_code(raw_text: str) -> str:
    """Extracts integer codes after 'DECISION:' tag or applies fallback keyword parsing."""
    if not raw_text or raw_text.strip() == "":
        return "0"

    match = re.search(r"DECISION:\s*([0-9,\s]+)", raw_text, re.IGNORECASE)
    if match:
        raw_codes = match.group(1)
        codes = [c.strip() for c in raw_codes.replace(" ", "").split(",") if c.strip().isdigit()]
        if codes:
            unique_codes = sorted(set(int(c) for c in codes))
            if len(unique_codes) > 1 and 0 in unique_codes:
                unique_codes.remove(0)
            return ", ".join(map(str, unique_codes))

    codes = set()
    if re.search(r"\b(discoloration|snail trail|tarnish|yellowing|golden|browning|patina)\b", raw_text, re.IGNORECASE):
        codes.add(5)
    if re.search(r"\b(delamination|void|peeling)\b", raw_text, re.IGNORECASE):
        codes.add(6)

    if codes:
        return ", ".join(map(str, sorted(codes)))
    return "0"


# ---------------------------------------------------------
# Inspection Protocol & Prompt
# ---------------------------------------------------------
PROMPT_TEXT = """Analyze this laboratory-captured, calibrated visible-light (VI) image of a solar cell for defects under flat illumination.

TARGET DEFECT TAXONOMY:
- Code 5 (Discoloration / Snail Trails / Tarnishing): Any yellowish, golden, brassy, or brownish tint along the silver grid fingers or horizontal busbars (including pale, smooth, or diffuse patina without dark borders), or localized staining on the wafer surface.
- Code 6 (Delamination): Milky, frosted optical voids or peeling along cell edges or busbars.
- Code 0 (Healthy): Uniform surface tone with intact grid lines and no chemical tracks, tarnish, or optical voids.

INSPECTION PROTOCOL:
1. Relative Busbar Comparison: Compare the hue of the middle horizontal busbar against the top and bottom busbars. If any busbar shows a warm, yellowish, golden, or brassy band—even if it is diffuse, smooth, or lacks sharp borders—classify as Code 5.
2. Grid & Surface: Check for jagged snail trails or localized wafer discoloration.
3. Cell Borders: Check for milky optical delamination voids.

OUTPUT FORMAT:
Provide 1 brief sentence stating the visual status of the busbars and cell surface, followed immediately by:
DECISION: <comma-separated codes, or 0 if healthy>"""


def classify_image(client: OpenAI, image_path: str) -> tuple[str, str]:
    """Classifies an image and returns (parsed_code, reasoning_text)."""
    try:
        data_uri = encode_file_to_data_uri(image_path)
    except Exception as e:
        return f"ERROR: {e}", ""

    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-32B-Instruct",
            temperature=0.0,
            max_tokens=150,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert industrial photovoltaic inspector specializing in visible-light defect classification."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                },
            ],
        )
        content = completion.choices[0].message.content or ""
        parsed_code = extract_decision_code(content)
        return parsed_code, content.strip().replace("\n", " ")

    except Exception as e:
        return f"API_ERROR: {e}", ""


# ---------------------------------------------------------
# Batch Pipeline
# ---------------------------------------------------------
def main():
    base_url = normalize_base_url(BASE_URL)
    if not SKIP_ENDPOINT_CHECK:
        endpoint_ok, endpoint_error = check_api_endpoint(
            base_url,
            timeout_seconds=HEALTHCHECK_TIMEOUT_SECONDS,
            retries=HEALTHCHECK_RETRIES,
        )
        if not endpoint_ok:
            print(f"ERROR: {endpoint_error}")
            return

    image_dir = Path(IMAGE_DIR)
    if not image_dir.is_dir():
        print(f"ERROR: Directory not found: {IMAGE_DIR}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions],
        key=lambda x: x.name.lower()
    )

    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return

    output_path = Path(OUTPUT_INTEGER_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    print(f"Saving classifications to: {OUTPUT_INTEGER_CSV}")

    # Standard OpenAI client automatically sends Authorization: Bearer <API_KEY>
    client = OpenAI(api_key=API_KEY, base_url=base_url)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "classification", "reasoning"])

        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Classifying {image_path.name}...", end=" ", flush=True)
            code, reasoning = classify_image(client, str(image_path))

            if code.startswith("API_ERROR:"):
                print(f"\n{code}")
                print("Aborting to avoid sending queries to a failing endpoint.")
                return

            print(f"Result -> {code}")
            writer.writerow([image_path.name, code, reasoning])

    print(f"\nProcessing complete: {output_path}")


if __name__ == "__main__":
    main()