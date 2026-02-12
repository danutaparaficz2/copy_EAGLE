import os
import base64
from dotenv import load_dotenv
import mimetypes
import csv
from pathlib import Path
from openai import OpenAI
from PIL import Image
import io
# Option 2: Load .env from a specific path
load_dotenv('/Users/eagle/Documents/.env')
# Directory containing images to classify
panel=os.getenv('PANEL', '23-P09-D')
IMAGE_DIR = (f"/Users/eagle/Documents/eagle-classification/normalized_images/{panel}/EL/")
# Output CSV file
OUTPUT_CSV = (f"/Users/eagle/Documents/eagle-classification/OPENAI/{panel}/classification_results_EL_{panel}.csv")
OUTPUT_INTEGER_CSV = (f"/Users/eagle/Documents/eagle-classification/OPENAI/{panel}/classification_results_EL_{panel}_integer.csv")
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"

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
        data_uri = encode_file_to_data_uri(image_path)
    except Exception as e:
        return f"ERROR: {e}"

    prompt_text = """
    Analyze the solar cell EL image.

    Only report defects that are clearly visible and unambiguous.

    Do NOT classify the following as defects:
    - normal vertical EL stripe patterns
    - uniform intensity variations
    - faint texture lines aligned with cell structure
    - busbar shadows
    - imaging noise

    Defect definitions:
    Crack: a clearly visible, continuous, irregular fracture line not aligned with gridlines.
    Cross: a broken gridline or busbar segment forming a short cross-like interruption.
    Dark: a large, clearly darker region significantly darker than surroundings.
    Corrosion: visible oxidation or discoloration near busbars or fingers.

    Return only:
    - good
    - Crack
    - Cross
    - Dark
    - Corrosion

    If no clear defect is present, return: good.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            top_p=1,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict industrial solar cell EL defect inspector. "
                            "Be conservative. Only report defects that are clearly visible and unambiguous. "
                            "If uncertain, return 'good'."
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
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return " ".join(t for t in texts if t.strip())
        else:
            return content or "No response"
    except Exception as e:
        return f"API_ERROR: {e}"

def main():
    if not API_KEY:
        print("Missing API_KEY environment variable.")
        return

    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"ERROR: Directory not found or not a directory: {IMAGE_DIR}")
        return

    # Collect all image files (common extensions)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
    image_files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Found {len(image_files)} image(s) in {IMAGE_DIR}")
    print(f"Results will be saved to {OUTPUT_CSV}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Open CSV and write results
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "classification"])

        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Classifying {image_path.name}...", end=" ")
            result = classify_image(client, str(image_path))
            print(f"Result: {result} for {image_path.name}")
            writer.writerow([image_path.name, result])
            print("Done.")

    print(f"\nClassification complete. Results saved to {OUTPUT_CSV} AND {OUTPUT_INTEGER_CSV}")

    convert_csv(OUTPUT_CSV, OUTPUT_INTEGER_CSV)



if __name__ == "__main__":
    main()