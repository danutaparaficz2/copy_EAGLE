from openai import OpenAI
import os
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
import io
from PIL import Image

# Option: Load .env from a specific path
load_dotenv('/Users/eagle/Documents/.env')
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def image_to_data_url(path: str) -> str:
    """
    Convert an image file to a base64 data URL for OpenAI API input.
    Assumes PNG; change MIME type if using JPEG/TIFF-converted images.
    """
    path = Path(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        # For TIFF or other scientific formats, convert to PNG/JPEG before sending.
        raise ValueError(f"Unsupported image type: {suffix}. Convert to PNG or JPEG first.")

    return f"data:{mime};base64,{b64}"

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

SYSTEM_PROMPT = """
You are assisting with prelabeling defects in photovoltaic solar cell images.

Each solar cell may be:
1. good / normal, with no visible defect evidence, or
2. defective, with one or more defect categories present.

You must classify the cell using three imaging modalities:
- VI: visible image
- EL: electroluminescence image
- UVf: ultraviolet fluorescence image

Possible defect categories:

1. crack
A solar cell crack is present. EL imaging reveals cracks most clearly. Depending on module materials and layer structure, the crack may also become visible in UVf at later stages. In a few cases, cracks can appear in VI, especially when encapsulant discoloration is present in non-cracked areas.

2. cross
Small cross-shaped cracks on solar cells. These cracks are typically very small and can easily be missed during human EL inspection. They are directly visible in EL. Over time, and depending on module materials and layer structure, they may also become visible in UVf due to oxygen bleaching.

3. dark
A dark region is present in the EL image of the solar cell. This phenomenon appears only under electroluminescence and should therefore be visible exclusively in EL images. The dark area may be caused by a disconnected cell region, damaged busbars, broken fingers, or similar defects.

4. corrosion
Corrosion is present on the solar cell. In some cases, corrosion can be identified in both EL and VI, while in others it is visible in only one of them, depending on where it occurs.

5. discoloration
Discoloration is present and is directly visible in VI images. Since discoloration is mostly caused by degradation by-products, these compounds may exhibit fluorescence, so discolored areas can appear brighter in UVf images. If degradation by-products lead to corrosion, for example acetic acid formed during EVA degradation, corrosion features may also become visible in EL at later stages.
Localized discoloration may be supported by UVf because UVf can identify affected regions. Uniform discoloration may make the whole UVf cell brighter, but cell-level UVf brightness alone is not sufficient for classification.

6. delamination
Delamination is present between at least two layers of the PV module stack. At early stages, delamination is typically detectable only in VI images. At later stages, if delaminated areas lead to secondary defects such as corrosion from moisture ingress, evidence may also appear in other modalities.

Important rules:
- This is a multi-label task. More than one defect may be present.
- Do not force a defect label. If evidence is weak, mark the defect as absent or uncertain.
- Use "normal" only when there is no convincing evidence for any listed defect.
- For each predicted defect, report the imaging modalities that support the decision.
- Distinguish "dark" from other defects: dark should be based on EL-only dark regions.
- Distinguish "cross" from ordinary crack: cross refers to small cross-shaped cracks, usually visible in EL.
- Return only valid JSON.
"""


def classify_pv_cell(
    cell_id: str,
    vi_path: str,
    el_path: str,
    uv_path: str,
    model: str = "gpt-4.1-mini"
) -> dict:
    """
    Classify one solar cell using VI, EL, and UVf images.
    Returns a Python dictionary parsed from the model's JSON output.
    """

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_PROMPT
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Classify solar cell {cell_id}. "
                            "The three images are provided in this order: VI, EL, UVf. "
                            "Return the multi-label defect assessment."
                        )
                    },
                    {
                        "type": "input_image",
                        #"image_url": image_to_data_url(vi_path)
                        "image_url": encode_file_to_data_uri(vi_path)
                    },
                    {
                        "type": "input_image",
                        "image_url": encode_file_to_data_uri(el_path)
                    },
                    {
                        "type": "input_image",
                        "image_url": encode_file_to_data_uri(uv_path)
                    }
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "eagle_pv_cell_multilabel",
                "schema": {
                    "type": "object",
                    "properties": {
                        "cell_id": {
                            "type": "string"
                        },
                        "overall_class": {
                            "type": "string",
                            "enum": ["normal", "defective", "uncertain"]
                        },
                        "defect_present": {
                            "type": "boolean"
                        },
                        "defects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "defect_type": {
                                        "type": "string",
                                        "enum": [
                                            "crack",
                                            "cross",
                                            "dark",
                                            "corrosion",
                                            "discoloration",
                                            "delamination"
                                        ]
                                    },
                                    "presence": {
                                        "type": "string",
                                        "enum": ["present", "absent", "uncertain"]
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "severity": {
                                        "type": "string",
                                        "enum": [
                                            "none",
                                            "low",
                                            "medium",
                                            "high",
                                            "uncertain"
                                        ]
                                    },
                                    "supporting_modalities": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": ["VI", "EL", "UVf"]
                                        }
                                    },
                                    "visual_evidence": {
                                        "type": "string"
                                    },
                                    "reasoning_note": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "defect_type",
                                    "presence",
                                    "confidence",
                                    "severity",
                                    "supporting_modalities",
                                    "visual_evidence",
                                    "reasoning_note"
                                ],
                                "additionalProperties": False
                            }
                        },
                        "final_labels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "normal",
                                    "crack",
                                    "cross",
                                    "dark",
                                    "corrosion",
                                    "discoloration",
                                    "delamination",
                                    "uncertain"
                                ]
                            }
                        },
                        "needs_human_review": {
                            "type": "boolean"
                        },
                        "human_review_reason": {
                            "type": "string"
                        },
                        "summary": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "cell_id",
                        "overall_class",
                        "defect_present",
                        "defects",
                        "final_labels",
                        "needs_human_review",
                        "human_review_reason",
                        "summary"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    result = json.loads(response.output_text)
    return result


# Example usage
if __name__ == "__main__":
    panel = '24-P10-A'
    print(f"Classifying cells in panel {panel}...")
    IMAGE_DIR = f"/Users/eagle/Documents/eagle-classification/normalized_images/{panel}/VI/"
    files = os.listdir(IMAGE_DIR)
    fname = files[0]
    cell_id=fname.split('.')[0].split('_')[-2]
    print(f"Classifying cell {cell_id}")
    result = classify_pv_cell(
        cell_id=cell_id,
        vi_path=os.path.join(IMAGE_DIR, fname),
        el_path=os.path.join(IMAGE_DIR.replace('VI', 'EL'), fname.replace('VI', 'EL')),
        uv_path=os.path.join(IMAGE_DIR.replace('VI', 'UV'), fname.replace('VI', 'UV'))
    )

    print(json.dumps(result, indent=2))