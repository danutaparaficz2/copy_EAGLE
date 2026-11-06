import base64
import csv
import io
import json
import os
from pathlib import Path
from typing import List, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError
import requests

# ---------------------------------------------------------
# Environment & Path Configuration
# ---------------------------------------------------------
load_dotenv("/Users/eagle/Documents/.env")

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "token-not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-32B-Instruct")

PANEL = os.getenv("PANEL", "24-P10-A")
IMAGE_DIR = f"/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/{PANEL}/EL/"
OUTPUT_INTEGER_CSV = f"/Users/danuta.paraficz/PyProjects/eagle-classification/OPENAI/{PANEL}/classification_results_EL_{PANEL}_agentic.csv"

HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("OPENAI_HEALTHCHECK_TIMEOUT", "15.0"))
HEALTHCHECK_RETRIES = max(1, int(os.getenv("OPENAI_HEALTHCHECK_RETRIES", "3")))
SKIP_ENDPOINT_CHECK = os.getenv("OPENAI_SKIP_ENDPOINT_CHECK", "0").strip().lower() in {"1", "true", "yes"}

DEFECT_TO_INT = {
    "good": 0,
    "crack": 1,
    "cross": 2,
    "dark": 3,
    "corrosion": 4,
}

# ---------------------------------------------------------
# Structured Schemas for Strict Decoding
# ---------------------------------------------------------
ValidDefectType = Literal["good", "crack", "cross", "dark", "corrosion"]


class InitialInspectionResult(BaseModel):
    is_defective: bool = Field(..., description="False if normal cell; True if clearly visible defect present.")
    defects: List[ValidDefectType] = Field(
        ...,
        description="List of detected defects: 'crack', 'cross', 'dark', 'corrosion'. Use ['good'] if none.",
    )
    salient_feature: str = Field(..., description="1-sentence description of the most prominent pattern observed.")


class DefectBox(BaseModel):
    category: ValidDefectType = Field(..., description="Defect type this box belongs to.")
    # Normalized coordinates [0 to 1000]
    ymin: int = Field(..., ge=0, le=1000)
    xmin: int = Field(..., ge=0, le=1000)
    ymax: int = Field(..., ge=0, le=1000)
    xmax: int = Field(..., ge=0, le=1000)


class CriticVerificationResult(BaseModel):
    confirmed_defective: bool = Field(
        ...,
        description="True ONLY if the defect is unambiguous, clearly visible, and NOT a busbar, finger grid, or corner chamfer.",
    )
    final_defects: List[ValidDefectType] = Field(
        ...,
        description="Verified defect classes. Return ['good'] if initial flag was an artifact, shadow, or normal structure.",
    )
    critic_rationale: str = Field(..., description="1-sentence verification justification.")
    boxes: List[DefectBox] = Field(
        default_factory=list,
        description="One bounding box per confirmed defect, normalized to [0-1000] as ymin, xmin, ymax, xmax. Empty if none confirmed.",
    )


# ---------------------------------------------------------
# Streamlined Prompts Optimized for 32B Parameter VLMs
# ---------------------------------------------------------
PROMPT_INITIAL_PASS = """Inspect this 250x250 electroluminescence (EL) image of a single solar cell.


NORMAL CELL STRUCTURES (DO NOT FLAG):
- Straight, continuous parallel grid lines (fingers)
- Wide horizontal/vertical contact strips (busbars)
- Symmetrical corner cutouts (chamfers)

CLEAR DEFECT DEFINITIONS:
- crack: Bold, jagged, or distinct continuous dark fracture path across the silicon (can run diagonally, vertically, or alongside/under busbars).
- cross: Two fracture lines intersecting in an X, +, or Y junction.
- dark: Prominent inactive, disconnected, or noticeably sub-luminescent zones (including partially darkened edges, or step-changes where one side is distinctly dimmer than healthy cell bulk).
- corrosion: Diffuse, hazy, low-contrast dark halos or fuzzy intensity drops along the busbars and contact fingers (non-sharp emission loss caused by contact resistance, distinct from sharp mechanical fracture lines).
- good: Defect-free, or exhibiting only normal structural lines and benign noise.

Output strict JSON matching the requested schema.
"""

PROMPT_CRITIC_PASS = """You are a senior photovoltaic QA inspector verifying candidate defects on this 250x250 EL solar cell.
A screening pass flagged this cell with: {candidate_labels}
Screening observation: {candidate_observation}

CHALLENGE TASK:
1. Is this candidate defect an actual, high-contrast, clearly visible fault?- Note on 'dark': Includes both completely dead (black) regions AND clearly delineated sub-luminescent (under-emitting) bands or segments.
2. Or is it merely a standard busbar, a grid finger artifact, normal background gradient, or corner edge?

Output strict JSON confirming or correcting the classification.
For every confirmed defect, also give a bounding box (ymin, xmin, ymax, xmax) normalized to [0-1000].
"""


# ---------------------------------------------------------
# Helpers & Client
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
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


def run_vlm_inference(client: OpenAI, model: str, data_uri: str, prompt: str, schema_class: type[BaseModel]) -> dict:
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                    {
                        "type": "text",
                        "text": f"{prompt}\nRespond ONLY with JSON matching this schema, including every required field:\n"
                        f"{json.dumps(schema_class.model_json_schema())}",
                    },
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_class.__name__,
                "schema": schema_class.model_json_schema(),
                "strict": True,
            },
        },
    )
    raw_content = completion.choices[0].message.content or "{}"
    return json.loads(raw_content)


def show_confirmed_boxes(image_path: str, boxes: List[DefectBox]) -> None:
    """Displays the cell with bounding boxes of confirmed defects drawn on it."""
    with Image.open(image_path) as src:
        img = src.convert("RGB")
    scale = max(1, 750 // max(img.size))
    img = img.resize((img.width * scale, img.height * scale))
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for b in boxes:
        box = (b.xmin / 1000 * w, b.ymin / 1000 * h, b.xmax / 1000 * w, b.ymax / 1000 * h)
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 4, box[1] + 4), b.category, fill="yellow")
    img.show(title=Path(image_path).name)


def labels_to_int_string(labels: List[str]) -> str:
    codes = set()
    for lbl in labels:
        lbl_clean = lbl.strip().lower()
        if lbl_clean in DEFECT_TO_INT:
            codes.add(DEFECT_TO_INT[lbl_clean])
    
    # If genuine defects exist, discard 'good' (0)
    if len(codes) > 1 and 0 in codes:
        codes.remove(0)

    if not codes:
        return "0"
    return ", ".join(map(str, sorted(codes)))


# ---------------------------------------------------------
# Agentic Inspection Pipeline
# ---------------------------------------------------------
def agentic_classify_cell(client: OpenAI, model: str, image_path: str) -> tuple[str, str, str]:
    """
    Executes a 2-stage agentic workflow:
    1. Fast initial inspection.
    2. If a defect is flagged, triggers an adversarial Critic to challenge false positives.
    """
    try:
        data_uri = encode_file_to_data_uri(image_path)
    except Exception as e:
        return f"FILE_ERROR: {e}", "", "FAILED"

    try:
        # Pass 1: Screening
        init_raw = run_vlm_inference(client, model, data_uri, PROMPT_INITIAL_PASS, InitialInspectionResult)
        init_res = InitialInspectionResult(**init_raw)

        # Quick exit path: High-confidence clean cell
        if not init_res.is_defective or "good" in init_res.defects:
            return "0", init_res.salient_feature, "CLEAN_AUTO_ACCEPTED"

        # Pass 2: Adversarial Critic (only for flagged anomalies)
        critic_prompt = PROMPT_CRITIC_PASS.format(
            candidate_labels=", ".join(init_res.defects),
            candidate_observation=init_res.salient_feature,
        )
        critic_raw = run_vlm_inference(client, model, data_uri, critic_prompt, CriticVerificationResult)
        critic_res = CriticVerificationResult(**critic_raw)

        if critic_res.confirmed_defective and "good" not in critic_res.final_defects:
            int_code = labels_to_int_string(critic_res.final_defects)
            reasoning = f"Verified {critic_res.final_defects}: {critic_res.critic_rationale}"
            flow_status = "DEFECT_CONFIRMED"
            if critic_res.boxes:
                show_confirmed_boxes(image_path, critic_res.boxes)
        else:
            int_code = "0"
            reasoning = f"Pruned false positive ({init_res.defects}): {critic_res.critic_rationale}"
            flow_status = "DEFECT_REJECTED_AS_ARTIFACT"

        return int_code, reasoning, flow_status

    except (ValidationError, json.JSONDecodeError) as e:
        return "PARSE_ERROR", str(e).replace("\n", " "), "PARSE_ERROR"
    except Exception as e:
        return f"API_ERROR: {e}", "", "ERROR"


# ---------------------------------------------------------
# Main Execution Loop
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

    print(f"Loaded {len(image_files)} cell images from: {IMAGE_DIR}")
    print(f"Writing agentic results to: {OUTPUT_INTEGER_CSV}")

    client = OpenAI(api_key=API_KEY, base_url=base_url)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "classification", "reasoning", "flow_status"])

        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_path.name}...", end=" ", flush=True)
            code, reasoning, flow_status = agentic_classify_cell(client, MODEL_NAME, str(image_path))

            if code.startswith("API_ERROR:"):
                print(f"\n{code}")
                print("Endpoint communication broken. Halting.")
                return

            print(f"-> Code: {code} [{flow_status}]")
            writer.writerow([image_path.name, code, reasoning, flow_status])
            csvfile.flush()

    print(f"\nDone. Results saved to: {output_path}")


if __name__ == "__main__":
    main()