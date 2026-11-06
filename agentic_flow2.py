import base64
import csv
import io
import json
import os
import re
from pathlib import Path
from typing import List, Literal, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
import requests

# ---------------------------------------------------------
# Environment & Path Configuration
# ---------------------------------------------------------
load_dotenv("/Users/eagle/Documents/.env")

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "token-not-needed")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-32B-Instruct")

PANEL = os.getenv("PANEL", "24-P10-A")
IMAGE_DIR = f"/Users/danuta.paraficz/PyProjects/eagle-classification/normalized_images/{PANEL}/EL/"
OUTPUT_INTEGER_CSV = f"/Users/danuta.paraficz/PyProjects/eagle-classification/OPENAI/{PANEL}/classification_results_EL_{PANEL}_agentic_zoom.csv"

OUTPUT_ROOT = Path("/Users/danuta.paraficz/PyProjects/eagle-classification/Qwen")
GOOD_DIR = OUTPUT_ROOT / "good"
DEFECTED_DIR = OUTPUT_ROOT / "defected"
LABEL_FONT_SIZE = int(os.getenv("LABEL_FONT_SIZE", "28"))

HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("OPENAI_HEALTHCHECK_TIMEOUT", "15.0"))
HEALTHCHECK_RETRIES = max(1, int(os.getenv("OPENAI_HEALTHCHECK_RETRIES", "3")))
SKIP_ENDPOINT_CHECK = os.getenv("OPENAI_SKIP_ENDPOINT_CHECK", "0").strip().lower() in {"1", "true", "yes"}

# Integer mapping for downstream evaluation
DEFECT_TO_INT = {
    "good": 0,
    "crack": 1,
    "dark": 3,
    "corrosion": 4,
}

# ---------------------------------------------------------
# Pydantic Schemas with Fault-Tolerant Defaults
# ---------------------------------------------------------
ValidDefectType = Literal["crack", "dark", "corrosion"]


class CandidateDefect(BaseModel):
    category: ValidDefectType = Field(
        ...,
        description="Candidate defect type: 'crack', 'dark', or 'corrosion'.",
    )
    # Normalized coordinates [0 to 1000]
    ymin: int = Field(..., ge=0, le=1000)
    xmin: int = Field(..., ge=0, le=1000)
    ymax: int = Field(..., ge=0, le=1000)
    xmax: int = Field(..., ge=0, le=1000)
    observation: str = Field(default="", description="Short visual rationale for flagging this region.")


class GlobalScreeningResult(BaseModel):
    is_defective: bool = Field(
        default=False,
        description="True if one or more clearly visible defects are found; False if completely clean.",
    )
    candidates: List[CandidateDefect] = Field(
        default_factory=list,
        description="List of detected defect candidates with bounding boxes.",
    )
    cell_summary: str = Field(
        default="Clean cell",
        description="1-sentence description of overall cell condition.",
    )


class CriticPatchVerification(BaseModel):
    is_genuine_defect: bool = Field(
        default=False,
        description="True ONLY if the zoomed crop reveals an undeniable defect.",
    )
    confirmed_category: Optional[ValidDefectType] = Field(
        default=None,
        description="Verified defect category. Set to null if rejected.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    critic_reasoning: str = Field(
        default="Verified by critic",
        description="Justification for confirming or rejecting the patch.",
    )


# ---------------------------------------------------------
# Domain-Accurate Prompts (Grayscale Intensity Grounded)
# ---------------------------------------------------------
PROMPT_GLOBAL_SCREENING = """You are inspecting a 250x250 grayscale electroluminescence (EL) image of a single solar cell.

TASK: Identify CLEARLY VISIBLE macro defects and localize each with a normalized bounding box [0-1000] (ymin, xmin, ymax, xmax).

DO NOT FLAG NORMAL STRUCTURAL FEATURES:
- Continuous straight parallel lines (contact grid fingers)
- Wide horizontal/vertical contact strips (busbars)
- Clean diagonal corner cutouts (chamfers)

DEFECT DEFINITIONS (GRAYSCALE INTENSITY):
- crack: Bold, jagged, or distinct continuous dark fracture path across the silicon (can run diagonally, vertically, or alongside/under busbars).
- dark: Prominent inactive, disconnected, or noticeably sub-luminescent zones (including partially darkened edges, or step-changes where one side is distinctly dimmer than healthy cell bulk).
- corrosion: Diffuse, hazy, low-contrast dark halos or fuzzy intensity drops along the busbars and contact fingers (non-sharp emission loss caused by contact resistance, distinct from sharp mechanical fracture lines).

If no prominent defects exist, set is_defective=false and candidates=[].
Output strict JSON matching the schema.
"""

PROMPT_ZOOMED_CRITIC = """You are a senior QA inspector conducting optical verification on a cropped sub-patch of a grayscale EL solar cell.
A screening pass flagged this region as candidate '{candidate_category}'.
Screening note: {candidate_observation}

VERIFICATION GUIDELINES:
1. CRACK / FRACTURE:
   - Any distinct, continuous dark line cutting across or alongside grid fingers or busbars is a REAL CRACK.
   - Do NOT dismiss a line just because it runs near or parallel to a busbar.
2. DARK / STEP-CHANGE:
   - A clear boundary where one side is distinctly brighter and the other side is noticeably dimmer/darker is an active defect (partial disconnection / carrier loss).
3. CORROSION:
   - Confirmed if there are diffuse, fuzzy dark halos or soft intensity drops around the busbars/fingers, not a sharp crack.
4. ONLY REJECT IF:
   - The feature is 100% strictly a normal straight vertical grid finger, normal horizontal busbar solder pad, or diagonal corner edge chamfer with no anomalous lines or contrast steps.

Output strict JSON: set is_genuine_defect=true if the anomaly is confirmed, and assign confirmed_category.
"""


# ---------------------------------------------------------
# Network & JSON Parsing Helpers
# ---------------------------------------------------------
def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    return normalized


def get_active_model_name(client: OpenAI, default_fallback: str) -> str:
    """Queries vLLM to find the exact registered model identifier to prevent 404 errors."""
    try:
        models = client.models.list()
        if models.data:
            detected = models.data[0].id
            print(f"Connected to vLLM. Active model detected: '{detected}'")
            return detected
    except Exception as e:
        print(f"Notice: Could not list models from endpoint ({e}). Using: '{default_fallback}'")
    return default_fallback


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


def pil_to_data_uri(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def clean_and_parse_json(raw_text: str) -> dict:
    """Extracts valid JSON even if wrapped in markdown formatting or preamble."""
    if not raw_text or not raw_text.strip():
        return {}

    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw_text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return {}


def run_vlm_structured(client: OpenAI, model: str, pil_img: Image.Image, prompt: str, schema_class: type[BaseModel]) -> dict:
    data_uri = pil_to_data_uri(pil_img)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=450,
        messages=[
            {
                "role": "system",
                "content": f"You are a strict automated inspection system. Output ONLY valid JSON matching this schema:\n{json.dumps(schema_class.model_json_schema())}",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    raw_content = completion.choices[0].message.content or "{}"
    return clean_and_parse_json(raw_content)


def crop_candidate_patch(full_img: Image.Image, candidate: CandidateDefect, margin_pct: float = 0.25) -> Image.Image:
    """
    Tool Action: Extracts a zoomed patch with sufficient surrounding context
    so the Critic can distinguish structural busbars from adjacent cracks.
    Enforces a minimum 100x100 px field-of-view.
    """
    w, h = full_img.size
    box_w = candidate.xmax - candidate.xmin
    box_h = candidate.ymax - candidate.ymin

    pad_x = max(25, int(box_w * margin_pct))
    pad_y = max(25, int(box_h * margin_pct))

    xmin = max(0, int(((candidate.xmin - pad_x) / 1000) * w))
    ymin = max(0, int(((candidate.ymin - pad_y) / 1000) * h))
    xmax = min(w, int(((candidate.xmax + pad_x) / 1000) * w))
    ymax = min(h, int(((candidate.ymax + pad_y) / 1000) * h))

    # Maintain at least 100x100 px resolution to prevent macro context loss
    if (xmax - xmin) < 100:
        diff = 100 - (xmax - xmin)
        xmin = max(0, xmin - diff // 2)
        xmax = min(w, xmax + (diff - diff // 2))
    if (ymax - ymin) < 100:
        diff = 100 - (ymax - ymin)
        ymin = max(0, ymin - diff // 2)
        ymax = min(h, ymin + (diff - diff // 2))

    return full_img.crop((xmin, ymin, xmax, ymax))


def save_good_image(full_img: Image.Image, image_path: Path) -> None:
    GOOD_DIR.mkdir(parents=True, exist_ok=True)
    full_img.save(GOOD_DIR / f"{image_path.stem}.png")


def save_defected_image(full_img: Image.Image, confirmed: list[tuple[str, CandidateDefect]], image_path: Path) -> None:
    """Saves the cell with bounding boxes and large category labels drawn on it."""
    scale = max(1, 750 // max(full_img.size))
    img = full_img.resize((full_img.width * scale, full_img.height * scale))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=LABEL_FONT_SIZE)
    except TypeError:
        font = ImageFont.load_default()
    w, h = img.size
    for category, cand in confirmed:
        box = (cand.xmin / 1000 * w, cand.ymin / 1000 * h, cand.xmax / 1000 * w, cand.ymax / 1000 * h)
        draw.rectangle(box, outline="red", width=3)
        text_xy = (box[0] + 4, box[1] + 4)
        text_box = draw.textbbox(text_xy, category, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text(text_xy, category, fill="yellow", font=font)
    DEFECTED_DIR.mkdir(parents=True, exist_ok=True)
    img.save(DEFECTED_DIR / f"{image_path.stem}.png")


def format_labels_to_csv_code(label_set: set[str]) -> str:
    """Maps confirmed categories to clean integer codes without undefined 9."""
    if not label_set or "good" in label_set:
        return "0"
    codes = sorted([DEFECT_TO_INT[lbl] for lbl in label_set if lbl in DEFECT_TO_INT and DEFECT_TO_INT[lbl] != 0])
    return ", ".join(map(str, codes)) if codes else "0"


# ---------------------------------------------------------
# Agentic Zoom & Crop Orchestrator
# ---------------------------------------------------------
def process_cell_with_zoom(client: OpenAI, model: str, image_path: Path) -> tuple[str, str, str]:
    try:
        full_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"FILE_ERROR: {e}", "", "ERROR"

    # Pass 1: Global Screening
    try:
        screening_raw = run_vlm_structured(client, model, full_img, PROMPT_GLOBAL_SCREENING, GlobalScreeningResult)
        screening = GlobalScreeningResult(**screening_raw)
    except Exception as e:
        return "0", f"Screening fallback: {e}", "SCREENING_FALLBACK_CLEAN"

    if not screening.is_defective or not screening.candidates:
        save_good_image(full_img, image_path)
        return "0", screening.cell_summary, "CLEAN_AUTO_ACCEPTED"

    confirmed_labels: set[str] = set()
    audit_notes = []
    confirmed_boxes: list[tuple[str, CandidateDefect]] = []

    # Iterate over candidates -> Tool Crop -> Zoomed Critic Pass
    for cand in screening.candidates:
        patch = crop_candidate_patch(full_img, cand)

        critic_prompt = PROMPT_ZOOMED_CRITIC.format(
            candidate_category=cand.category,
            candidate_observation=cand.observation,
        )
        try:
            critic_raw = run_vlm_structured(client, model, patch, critic_prompt, CriticPatchVerification)
            critic = CriticPatchVerification(**critic_raw)
        except Exception as e:
            audit_notes.append(f"Critic parse skipped ({cand.category}): {e}")
            continue

        if critic.is_genuine_defect and critic.confirmed_category:
            confirmed_labels.add(critic.confirmed_category)
            confirmed_boxes.append((critic.confirmed_category, cand))
            audit_notes.append(f"Confirmed [{critic.confirmed_category}] at [{cand.ymin},{cand.xmin},{cand.ymax},{cand.xmax}]: {critic.critic_reasoning}")
        else:
            audit_notes.append(f"Pruned [{cand.category}] artifact at [{cand.ymin},{cand.xmin},{cand.ymax},{cand.xmax}]: {critic.critic_reasoning}")

    if confirmed_boxes:
        save_defected_image(full_img, confirmed_boxes, image_path)
    else:
        save_good_image(full_img, image_path)

    final_code = format_labels_to_csv_code(confirmed_labels)
    combined_reasoning = " | ".join(audit_notes) if audit_notes else screening.cell_summary
    final_status = "DEFECT_CONFIRMED" if final_code != "0" else "ALL_CANDIDATES_REJECTED_AS_ARTIFACTS"

    return final_code, combined_reasoning, final_status


# ---------------------------------------------------------
# Main Execution Entry
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

    print(f"Processing {len(image_files)} cells with Zoom & Crop agentic loop...")
    print(f"Output CSV: {OUTPUT_INTEGER_CSV}")

    client = OpenAI(api_key=API_KEY, base_url=base_url)
    active_model = get_active_model_name(client, DEFAULT_MODEL)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "classification", "reasoning", "flow_status"])

        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_path.name}...", end=" ", flush=True)
            code, reasoning, flow_status = process_cell_with_zoom(client, active_model, image_path)

            if code.startswith("API_ERROR:") or code.startswith("FILE_ERROR:"):
                print(f"\n{code}")
                return

            print(f"-> Code: {code} [{flow_status}]")
            writer.writerow([image_path.name, code, reasoning, flow_status])
            csvfile.flush()

    print(f"\nFinished processing all images. Results written to: {output_path}")


if __name__ == "__main__":
    main()