import argparse
import os
import base64
import csv
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv('myenv/.env')
LOCAL_API_KEY = "token-not-needed"
LOCAL_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
load_dotenv('/Users/eagle/Documents/env_n/.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are an industrial solar cell inspector analyzing multimodal images.
You will receive three images of the same solar cell in this order: VI (visible light), EL (electroluminescence), UV (ultraviolet fluorescence).
You are assisting with prelabeling defects in photovoltaic solar cell images.

Each solar cell may be:
- good / normal, with no visible defect evidence, or
- defective, with one or more defect categories present.

You must classify the cell using three imaging modalities:
- VI: visible image
- EL: electroluminescence image
- UV: ultraviolet fluorescence image

Possible defect categories:

* crack
A solar cell crack is present. **Cracks can mainly be seen in EL mode.** EL imaging reveals cracks most clearly — look for dark lines, microcracks, or broken cell areas in the EL image first. Depending on module materials and layer structure, the crack may also become visible in UVf at later stages. In a few cases, cracks can appear in VI, especially when encapsulant discoloration is present in non-cracked areas. Do NOT classify as crack based on VI alone unless there is corroborating evidence in EL.

* cross
Small cross-shaped cracks on solar cells. These cracks are typically very small and can easily be missed during human EL inspection. They are directly visible in EL. Over time, and depending on module materials and layer structure, they may also become visible in UVf due to oxygen bleaching.

* dark
A dark region is present in the EL image of the solar cell. This phenomenon appears only under electroluminescence and should therefore be visible exclusively in EL images. The dark area may be caused by a disconnected cell region, damaged busbars, broken fingers, or similar defects.

* corrosion
Corrosion is present on the solar cell. In some cases, corrosion can be identified in both EL and VI, while in others it is visible in only one of them, depending on where it occurs.

* discoloration
Discoloration is present and is directly visible in VI images. Since discoloration is mostly caused by degradation by-products, these compounds may exhibit fluorescence, so discolored areas can appear brighter in UV images. If degradation by-products lead to corrosion, for example acetic acid formed during EVA degradation, corrosion features may also become visible in EL at later stages.
Localized discoloration may be supported by UV because UV can identify affected regions. Uniform discoloration may make the whole UV cell brighter, but cell-level UV brightness alone is not sufficient for classification.

* delamination
Delamination is present between at least two layers of the PV module stack. At early stages, delamination is typically detectable only in VI images. At later stages, if delaminated areas lead to secondary defects such as corrosion from moisture ingress, evidence may also appear in other modalities.

RULES:
- This is a multi-label task. More than one defect may be present.
- Do not force a defect label. If evidence is weak, mark the defect as absent or uncertain.
- Use "normal" only when there is no convincing evidence for any listed defect.
- Distinguish "dark" from other defects: dark should be based on EL-only dark regions.
- Distinguish "cross" from ordinary crack: cross refers to small cross-shaped cracks, usually visible in EL.

DEFECT DEFINITIONS:
- good: no visible defect evidence
- crack: crack visible MAINLY in EL (dark lines/microcracks in EL image); do NOT classify as crack from VI or UV alone
- cross: small cross-shaped crack, visible in EL
- dark: dark region in EL only (disconnected region, broken fingers, damaged busbars)
- corrosion: corrosion visible in EL and/or VI
- discoloration: color change visible in VI, possibly brighter in UV
- delamination: layer separation visible in VI, later in other modalities

OUTPUT FORMAT:
Return only the detected label names separated by spaces. Use lowercase.
Examples:
  good
  crack
  crack dark
  discoloration delamination
  crack corrosion discoloration

Do not return anything else."""
"""


RULES:
- This is a multi-label task. More than one defect may be present.
- Only classify a defect if clearly visible and structurally significant.
- Use "good" only when there is no convincing evidence for any defect.

OUTPUT FORMAT:
Return only the detected label names separated by spaces. Use lowercase.
Examples:
  good
  crack
  crack dark
  discoloration delamination
  crack corrosion discoloration

Do not return anything else."""


def encode_file_to_data_uri(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def classify_image(client: OpenAI, vi_path: str, el_path: str, uv_path: str, model_name: str = "nvidia/Qwen2.5-VL-7B-Instruct-NVFP4") -> str:
    """Classify a single cell from VI, EL, UV images and return space-separated text labels."""
    try:
        print("  [encoding]", end="", flush=True)
        start = time.time()
        vi_uri = encode_file_to_data_uri(vi_path)
        el_uri = encode_file_to_data_uri(el_path)
        uv_uri = encode_file_to_data_uri(uv_path)
        print(f" ✓ ({time.time()-start:.1f}s)", end="", flush=True)
    except Exception as e:
        return f"ERROR: {e}"

    try:
        print(" [inferring]", end="", flush=True)
        start = time.time()
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            top_p=1,
            timeout=300,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image 1 (VI — visible light):"},
                        {"type": "image_url", "image_url": {"url": vi_uri}},
                        {"type": "text", "text": "Image 2 (EL — electroluminescence, use this to detect cracks, dark regions, cross defects):"},
                        {"type": "image_url", "image_url": {"url": el_uri}},
                        {"type": "text", "text": "Image 3 (UV — ultraviolet fluorescence):"},
                        {"type": "image_url", "image_url": {"url": uv_uri}},
                        {"type": "text", "text": "Classify this solar cell using all three images. Cracks are mainly visible in EL. Return only label names separated by spaces."},
                    ],
                },
            ],
        )
        content = completion.choices[0].message.content
        elapsed = time.time() - start
        print(f" ✓ ({elapsed:.1f}s)", flush=True)
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return " ".join(t for t in texts if t.strip())
        return content.strip() if content else "good"
    except Exception as e:
        print(f" ✗ ERROR", flush=True)
        return f"API_ERROR: {e}"


def main():
    print("[START] classify_llm.py starting up...", flush=True)

    parser = argparse.ArgumentParser(description="LLM-based PV cell classification (VI+EL+UVf)")
    parser.add_argument("--panel", default="24-P10-A", type=str, help="Panel name, e.g. 24-P10-A")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API instead of local LLM")
    args = parser.parse_args()
    panel = args.panel
    use_openai = args.openai
    print(f"[INFO] Using panel: {panel}", flush=True)
    print(f"[INFO] Provider: {'OpenAI' if use_openai else 'Local LLM'}", flush=True)

    vi_dir = Path(f"normalized_images/{panel}/VI")
    el_dir = Path(f"normalized_images/{panel}/EL")
    uv_dir = Path(f"normalized_images/{panel}/UV")
    output_folder = Path("OPENAI") / panel
    label = "OLLM" if use_openai else "LLM"
    output_csv = output_folder / f"classification_results_{label}_{panel}.csv"
    output_folder.mkdir(parents=True, exist_ok=True)

    if not vi_dir.exists():
        print(f"ERROR: VI directory not found: {vi_dir}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    vi_files = sorted(
        [f for f in vi_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions],
        key=lambda x: x.name
    )

    if not vi_files:
        print(f"No VI images found in {vi_dir}")
        return

    print(f"Found {len(vi_files)} image(s) in {vi_dir}", flush=True)
    print(f"Results will be saved to {output_csv}", flush=True)

    # Resume: load already-done filenames from existing output
    done = set()
    if output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    done.add(row[0].strip())
        if done:
            print(f"[INFO] Resuming: {len(done)} already done, skipping them.", flush=True)

    vi_files_todo = [f for f in vi_files if f.name not in done]
    if not vi_files_todo:
        print("All images already classified. Nothing to do.")
        return

    if use_openai:
        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY not set in myenv/.env")
            return
        client = OpenAI(api_key=OPENAI_API_KEY)
        model_name = OPENAI_MODEL
        print(f"[INFO] Using OpenAI model: {model_name}", flush=True)
    else:
        client = OpenAI(api_key=LOCAL_API_KEY, base_url=LOCAL_BASE_URL)
        model_name = "nvidia/Qwen2.5-VL-7B-Instruct-NVFP4"
        print(f"[INFO] Client initialized with BASE_URL: {LOCAL_BASE_URL}", flush=True)

    # Append if resuming, write fresh otherwise
    file_mode = "a" if done else "w"
    with open(output_csv, file_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not done:
            writer.writerow(["filename", "classification"])

        total = len(vi_files)
        for i, vi_path in enumerate(vi_files_todo, 1):
            el_path = el_dir / vi_path.name.replace("VI", "EL")
            uv_path = uv_dir / vi_path.name.replace("VI", "UV")

            done_count = len(done) + i - 1
            print(f"[{done_count + 1}/{total}] {vi_path.name}", end="")
            sys.stdout.flush()

            result = classify_image(client, str(vi_path), str(el_path), str(uv_path), model_name)
            print(f" → {result}")
            sys.stdout.flush()

            writer.writerow([vi_path.name, result])
            csvfile.flush()

    print(f"\nClassification complete. Results saved to {output_csv}")


if __name__ == "__main__":
    main()