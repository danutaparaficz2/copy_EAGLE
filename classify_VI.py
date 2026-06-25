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

print("[MODULE] Starting imports...", file=sys.stderr, flush=True)

# Load .env from a specific path
print("[MODULE] Loading .env...", file=sys.stderr, flush=True)
load_dotenv('myenv/.env')
print("[MODULE] .env loaded", file=sys.stderr, flush=True)

#API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = "token-not-needed"
print(f"[MODULE] API_KEY loaded: {bool(API_KEY)}", file=sys.stderr, flush=True)

#BASE_URL = "https://api.openai.com/v1"
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
print(f"[MODULE] BASE_URL: {BASE_URL}", file=sys.stderr, flush=True)

import csv
import re
from pathlib import Path

# Label to integer mapping
LABEL_MAP = {
    'good': 0,
    'crack': 1,
    'cross': 2,
    'dark': 3,
    'corrosion': 4,
    'discoloration': 5,
    'delamination': 6
}
INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def parse_classification(text):
    """
    Parse classification text and return list of integer codes.
    
    Handles various formats:
    - "Good"
    - "Dark, Crack"
    - "Damaged: Crack, Dark"
    - "Damaged – Crack (fracture lines)"
    """
    if not text or text.strip() == '':
        return []
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Find all matching labels
    found_labels = []
    for label, code in LABEL_MAP.items():
        if label in text_lower:
            found_labels.append(code)
    
    # Remove duplicates and sort
    found_labels = sorted(set(found_labels))
    
    return found_labels


def convert_integer_csv_to_labels(input_file, output_file):
    """
    Convert integer-coded classification CSV back to string labels.
    E.g. '5 6' -> 'discoloration delamination', '0' -> 'good'
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    processed_rows = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0]
            codes_str = row[1].strip()
            labels = []
            for token in codes_str.split():
                try:
                    code = int(token)
                    labels.append(INT_TO_LABEL.get(code, str(code)))
                except ValueError:
                    labels.append(token)
            processed_rows.append((filename, ' '.join(labels) if labels else 'good'))
    processed_rows.sort(key=lambda r: r[0].lower())
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for filename, label_str in processed_rows:
            writer.writerow([filename, label_str])
    print(f"\u2713 Converted integer CSV to labels: {output_file}")


def convert_csv(input_file, output_file):
    """
    Convert classification CSV from text labels to integer codes.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    rows_processed = 0
    
    # Collect all processed rows first, then sort by filename
    processed_rows = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0]
            classification_text = row[1]
            codes = parse_classification(classification_text)
            codes_str = ','.join(map(str, codes)) if codes else ''
            processed_rows.append((filename, codes_str))
    # Sort rows by filename (case-insensitive)
    processed_rows.sort(key=lambda r: r[0].lower())
    rows_processed = len(processed_rows)

    # Write out sorted rows
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for filename, codes_str in processed_rows:
            writer.writerow([filename, codes_str])
    
    print(f"✓ Converted {rows_processed} rows")
    print(f"✓ Input: {input_file}")
    print(f"✓ Output: {output_file}")
    print(f"\nLabel mapping used:")
    for label, code in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        print(f"  {label.capitalize()}: {code}")



def encode_file_to_data_uri(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Open image with PIL
    img = Image.open(path)
    
    # Convert to RGB if needed (handles RGBA, L, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Convert to PNG format in memory
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
       
    return f"data:image/png;base64,{b64}"


def classify_image(client: OpenAI, image_path: str) -> str:
    """Classify a single image and return the assessment text."""
    try:
        print("  [encoding image]", end="", flush=True)
        start = time.time()
        data_uri = encode_file_to_data_uri(image_path)
        print(f" ✓ ({time.time()-start:.1f}s)", end="", flush=True)
    except Exception as e:
        return f"ERROR: {e}"

    # prompt_text = (
    #     "Analyze the solar cell visual band image to determine whether it appears good or damaged. "
    #     "If damaged, classify defects:  Discoloration -  clear change of color along buslines write 5, Delamination write 6. "
    #     "If good, write 0"
    #     "Return a concise assessment. Multiple defect types are possible."
    #     " Photo for each cell was made with very good light distribution. ")
    
    # prompt_text = """
    # Analyze the provided solar cell visible-light image.

    # INSPECTION RULES (strict):

    # 1. Only classify a defect if it is clearly visible and structurally significant.
    # 2. Ignore:
    # - minor lighting reflections
    # - small dust particles
    # - uniform shading variations
    # - camera noise
    # - minor surface texture variations

    # DEFECT DEFINITIONS:

    # 5 – Discoloration:
    # - A clear and continuous color change
    # - Located along busbars or gridlines
    # - Structurally different from surrounding material
    # - Not caused by lighting reflection

    # 6 – Delamination:
    # - Visible separation or lifting of layers
    # - Bubbling, peeling, or detachment
    # - Clear structural surface disruption

    # DECISION POLICY:
    # If the feature could reasonably be caused by lighting, reflection, or minor surface variation, classify as 0.

    # OUTPUT FORMAT:
    # Return only numbers separated by comma if multiple defects are clearly visible:
    # - 0 (good)
    # - 5
    # - 6

    # If no clear defect is present, return: 0
    # """


    prompt_text = """
    Analyze the provided solar cell visible-light image.

    INSPECTION RULES (strict):

    1. Only classify a defect if it is clearly visible and structurally significant.
    2. Ignore:
    - minor lighting reflections
    - small dust particles
    - uniform shading variations
    - camera noise
    - minor surface texture variations

    DEFECT DEFINITIONS:

    5 – Discoloration:
    - A clear and continuous color change
    - Located along busbars or gridlines
    - Structurally different from surrounding material
    - Not caused by lighting reflection

    6 – Delamination:
    - Visible separation or lifting of layers
    - Bubbling, peeling, or detachment
    - Clear structural surface disruption

    DECISION POLICY:
    If the feature could reasonably be caused by lighting, reflection, or minor surface variation, classify as 0.

    OUTPUT FORMAT:
    Return only numbers separated by space if multiple defects are clearly visible:
    - 0 (good)
    - 5 (Discoloration)
    - 6 (Delamination)

    If no clear defect is present, return: 0
    """
    try:
        print(" [inferring]", end="", flush=True)
        start = time.time()
        completion = client.chat.completions.create(
          #  model="gpt-4o",
            model="nvidia/Qwen2.5-VL-7B-Instruct-NVFP4",
            temperature=0,
            top_p=1,
            timeout=300,
            messages=[
                {
                    "role": "system",
                    "content":     "You are an industrial solar cell inspector analyzing visible light images. "
  
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt_text},
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
        else:
            return content or "No response"
    except Exception as e:
        print(f" ✗ ERROR", flush=True)
        return f"API_ERROR: {e}"

def main():
    print("[START] classify_VI.py starting up...", flush=True)
    
    parser = argparse.ArgumentParser(description="Classify VI solar cell images.")
    parser.add_argument("--panel", default="23-P09-B",
                        help="Panel identifier, e.g. 23-P09-B")
    args = parser.parse_args()
    print(f"[INFO] Using panel: {args.panel}", flush=True)

    panel = args.panel
    image_dir_str = f"normalized_images/{panel}/VI/"
    output_folder = os.path.join("OPENAI", panel)
    output_integer_csv = os.path.join(output_folder, f"classification_results_VI_{panel}_integer.csv")
    output_csv = os.path.join(output_folder, f"classification_results_VI_{panel}.csv")
    os.makedirs(output_folder, exist_ok=True)

    image_dir = Path(image_dir_str)
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"ERROR: Directory not found or not a directory: {image_dir_str}")
        return

    # Collect all image files (common extensions)
    print(f"[INFO] Scanning for images in: {image_dir_str}", flush=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
    image_files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {image_dir_str}")
        return

    print(f"Found {len(image_files)} image(s) in {image_dir_str}", flush=True)
    print(f"Results will be saved to {output_integer_csv}", flush=True)
    sorted_image_files = sorted(image_files, key=lambda x: x.name)

    print(f"[INFO] Initializing OpenAI client with BASE_URL: {BASE_URL}", flush=True)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"[INFO] Client initialized successfully", flush=True)

    # Open CSV and write results
    with open(output_integer_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "classification"])

        for i, image_path in enumerate(sorted_image_files, 1):
            print(f"[{i}/{len(sorted_image_files)}] {image_path.name}", end="")
            sys.stdout.flush()
            result = classify_image(client, str(image_path))
            print(f" → {result}")
            sys.stdout.flush()
            writer.writerow([image_path.name, result])
            csvfile.flush()

    print(f"\nClassification complete. Results saved to {output_integer_csv}")

    convert_integer_csv_to_labels(output_integer_csv, output_csv)
    print(f"Label CSV saved to {output_csv}")


if __name__ == "__main__":
    main()