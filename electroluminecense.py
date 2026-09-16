import base64
import csv
import io
import os
import re
from pathlib import Path
from urllib.parse import urlparse

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

PANEL = os.getenv("PANEL", "24-P10-A")
IMAGE_DIR = f"/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/{PANEL}/EL/"
OUTPUT_INTEGER_CSV = f"/Users/danuta.paraficz/PyProjects/eagle-classification/OPENAI/{PANEL}/classification_results_EL_{PANEL}_integer.csv"

HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("OPENAI_HEALTHCHECK_TIMEOUT", "15.0"))
HEALTHCHECK_RETRIES = max(1, int(os.getenv("OPENAI_HEALTHCHECK_RETRIES", "3")))
SKIP_ENDPOINT_CHECK = os.getenv("OPENAI_SKIP_ENDPOINT_CHECK", "0").strip().lower() in {"1", "true", "yes"}

# Integer mapping for downstream evaluation
DEFECT_TO_INT = {
    "good": 0,
    "crack": 1,
    "cross": 2,
    "dark": 3,
    "corrosion": 4,
    "undefined": 9,
}

# ---------------------------------------------------------
# Prompt Definition
# ---------------------------------------------------------
PROMPT_TEXT = """
You are inspecting one photovoltaic solar cell in an electroluminescence (EL) image.

INSPECTION PROTOCOL:

1. Inspect the entire cell for abnormal dark regions.
2. Inspect the silicon area for thin, irregular fracture lines.
3. Inspect intersections and branching crack structures.
4. Inspect busbars for diffuse/fuzzy dark regions consistent with
   corrosion.
5. Distinguish genuine defects from normal EL texture, busbars,
   fingers, borders, and corner chamfers.

Determine first whether the cell is:

GOOD
or
DEFECT
DEFECT DEFINITIONS

CRACK:
 irregular/jagged line, It may extend across a substantial portion of the cell.

CROSS:
A localized intersection of two or more crack-like fracture lines,
forming approximately a +, X, Y, or branching pattern.
A Cross is a localized crack intersection, not simply a normal crossing
of busbars or cell structure.

DARK:
A clearly abnormal dark/black inactive region.
The region must be substantially darker than the surrounding healthy
cell and must represent an actual inactive area, not normal EL texture.

CORROSION:
Diffuse, fuzzy, irregular darkening associated  along a busbar/contact region.
It is different from a sharp mechanical crack.

Return:
OBSERVATION: <one sentence>
DECISION: <"good" if defect-free, or a comma-separated list of detected classes: Crack, Cross, Dark, Corrosion>
"""


# ---------------------------------------------------------
# Utilities & Healthchecks
# ---------------------------------------------------------
def normalize_base_url(base_url: str) -> str:
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)

    print(
        f"{Path(path).name}: "
        f"mode={img.mode}, size={img.size}"
    )

    if img.mode != "RGB":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_decision_code(raw_text: str) -> str:
    """
    Parses textual class labels from the DECISION line and maps them to clean integer codes.
    Handles both names ('Crack, Dark') and unexpected raw numeric outputs.
    """
    if not raw_text or not raw_text.strip():
        return "0"

    match = re.search(r"DECISION:\s*([^\n\r]+)", raw_text, re.IGNORECASE)
    decision_str = match.group(1).strip() if match else raw_text

    detected_codes = set()
    tokens = [t.strip().lower() for t in re.split(r"[,;/]+", decision_str) if t.strip()]

    for token in tokens:
        if token in DEFECT_TO_INT:
            detected_codes.add(DEFECT_TO_INT[token])
        elif token.isdigit():
            detected_codes.add(int(token))
        else:
            for name, code in DEFECT_TO_INT.items():
                if re.search(rf"\b{name}\b", token):
                    detected_codes.add(code)

    # If genuine defects exist, remove good (0)
    if len(detected_codes) > 1 and 0 in detected_codes:
        detected_codes.remove(0)

    if detected_codes:
        return ", ".join(map(str, sorted(detected_codes)))
    return "0"


def classify_image(client: OpenAI, image_path: str) -> tuple[str, str]:
    """Sends image to vLLM/OpenAI endpoint and returns (integer_codes, cleaned_raw_text)."""
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
                    "content": "You are an expert industrial photovoltaic inspector specializing in electroluminescence (EL) defect classification.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
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
# Batch Execution
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
        key=lambda x: x.name.lower(),
    )

    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return

    output_path = Path(OUTPUT_INTEGER_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    print(f"Saving classifications to: {OUTPUT_INTEGER_CSV}")

    client = OpenAI(api_key=API_KEY, base_url=base_url)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "classification", "reasoning"])

        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_path.name}...", end=" ", flush=True)
            code, reasoning = classify_image(client, str(image_path))

            if code.startswith("API_ERROR:"):
                print(f"\n{code}")
                print("Aborting to avoid sending queries to a failing endpoint.")
                return

            print(f"Result -> {code} | Observation -> {reasoning}")
            writer.writerow([image_path.name, code, reasoning])
            csvfile.flush()

    print(f"\nProcessing complete: {output_path}")


if __name__ == "__main__":
    main()