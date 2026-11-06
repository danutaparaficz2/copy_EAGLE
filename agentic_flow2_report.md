# Report: `agentic_flow2.py` – Zoom & Crop Agentic Flow for EL Solar-Cell Defect Classification

## 1. Purpose

`agentic_flow2.py` classifies electroluminescence (EL) images of single solar cells (grayscale `.tif`) into
defect classes using a vision-language model (VLM, default `Qwen/Qwen3-VL-32B-Instruct`) served through an
OpenAI-compatible endpoint (vLLM).

| Class | Code |
|-------|------|
| good | 0 |
| crack | 1 |
| dark | 3 |
| corrosion | 4 |

(Code 2, formerly `cross`, has been removed; the remaining codes keep their numbers.)

Instead of asking the model for one verdict on the whole image, the script runs a **multi-step agentic loop**:
screen globally, crop each suspicious region, and let a second "critic" pass verify the zoomed crop.

## 2. The agentic flow

```
 cell image (.tif → RGB)
        │
        ▼
┌─────────────────────────────┐
│ Pass 1: Global Screening    │  VLM sees the whole cell
│  → is_defective             │  Outputs candidate defects, each with
│  → candidates[] (category,  │  a bounding box in normalized [0-1000]
│    bbox, observation)       │  coordinates
└──────────────┬──────────────┘
               │
     no candidates? ──yes──► code 0, saved to Qwen/good
               │no
               ▼
┌─────────────────────────────┐
│ Tool action: Crop & Zoom    │  Pure code (no model): cut the candidate
│  crop_candidate_patch()     │  bbox + padding (≥25/1000 or 25 % of box)
└──────────────┬──────────────┘
               │  one crop per candidate
               ▼
┌─────────────────────────────┐
│ Pass 2: Zoomed Critic       │  VLM sees ONLY the crop, plus the
│  → is_genuine_defect        │  candidate category and the screening note.
│  → confirmed_category       │  Decides: real defect or false alarm
│  → confidence, reasoning    │  (grid finger, busbar, chamfer, ...)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Aggregation                 │  Confirmed categories → integer code(s)
│                             │  e.g. "1, 3". None confirmed → 0
└──────────────┬──────────────┘
               ▼
   CSV row (filename, classification, reasoning, flow_status)
   + annotated image → Qwen/defected  (boxes + large labels)
   + or clean image  → Qwen/good
```

### Step details

1. **Global screening.** The prompt defines what is normal structure (grid fingers, busbars, corner chamfers)
   and what counts as a defect (crack, dark, corrosion). The model returns strict JSON validated with
   Pydantic (`GlobalScreeningResult`). If it finds nothing, the cell is accepted as good immediately
   (`CLEAN_AUTO_ACCEPTED`).
2. **Crop (tool use).** For every candidate, `crop_candidate_patch()` converts the normalized box to pixels,
   adds context padding so the critic can tell a crack from a neighbouring busbar, and guards against
   zero-area crops.
3. **Zoomed critic.** The crop is sent in a separate model call with a verification prompt
   (`PROMPT_ZOOMED_CRITIC`). The critic returns `is_genuine_defect`, `confirmed_category`, `confidence`
   and a short reasoning (`CriticPatchVerification`). Note that it can *re-assign* the category, not only
   accept or reject.
4. **Aggregation.** Confirmed categories are mapped to integer codes. Several defects in one cell give a
   comma-separated code (e.g. `1, 3`). If every candidate is rejected the cell is `0`
   (`ALL_CANDIDATES_REJECTED_AS_ARTIFACTS`).
5. **Outputs.** A CSV with the audit trail (every confirmed/pruned region with its coordinates and reasoning),
   plus images: good cells in `Qwen/good`, defective cells with red bounding boxes and large category labels
   in `Qwen/defected`.

### Robustness measures

- Tolerant JSON parsing (`clean_and_parse_json`) and defaulted schema fields, so a malformed reply does not
  crash the run.
- Screening failure falls back to `0` with status `SCREENING_FALLBACK_CLEAN`; a failed critic call skips that
  candidate and records it in the reasoning.
- The active model name is read from the server (`/models`) to avoid 404 errors from a wrong `MODEL_NAME`.
- `temperature=0` for deterministic output.

## 3. The "lost in the middle" problem

### What it is

"Lost in the middle" (Liu et al., 2023) describes the observation that language models use information
best when it is at the **beginning or end** of their input and worse when the relevant piece sits in the
**middle** of a long context. Performance follows a U-shaped curve over position.

The same family of effects appears with vision-language models: an image is converted into hundreds or
thousands of visual tokens. A small, low-contrast detail (such as a thin crack) is a tiny fraction of those
tokens, surrounded by many tokens of irrelevant structure. Attention is spread thinly, the fine detail is
diluted ("lost" among the other tokens), and the model tends to answer from the dominant global pattern. In
addition, image encoders downscale or tile large inputs, which can wash out fine, thin features. In our data
a single cell is about 1176×1178 px, while the defects (hairline cracks, faint sub-luminescent bands,
hazy corrosion halos) are small and low in contrast, making them prime candidates for being overlooked.

### How the agentic flow mitigates it

| Mechanism in `agentic_flow2.py` | Effect |
|---------------------------------|--------|
| **Crop & zoom** of each candidate region | The suspicious area fills the whole visual input, so the defect is no longer a small share of the tokens. It moves from "somewhere in the middle of a large image" to the entire context. |
| **Separate, short, focused critic call** | The critic gets one image and one question, with the candidate category and note placed right next to the image. There is no long context in which evidence can drift to the middle. |
| **Two-stage decomposition** (screen → verify) | Each stage solves a simpler task: stage 1 only has to *localize*, stage 2 only has to *judge*. Errors of one pass can be corrected by the other. |
| **One call per candidate** | Multiple defects are not competing for attention in a single answer. Each gets its own full-attention look. |
| **Padding around the crop** | Keeps enough neighbouring structure (busbars, fingers) so zooming does not remove the context needed for a correct judgement. |
| **Structured JSON output** | Forces an explicit decision on each field (category, box, confidence, reasoning) rather than a vague overall impression. |

In short: instead of hoping the model finds a needle in one large "haystack" image, the pipeline first
narrows the haystack down and then looks at the needle alone.

## 4. Limitations and caveats

- **Stage 1 is still global.** If the global screening misses a faint defect, the cell is accepted as good
  and the zoom stage never runs. Zooming reduces dilution only for regions that were flagged. A sliding-window
  or tiled screening pass would address this.
- **Bounding-box accuracy.** VLMs localize imprecisely. A wrong box produces a crop that misses the defect,
  and the critic may reject a real defect. Padding mitigates but does not eliminate this.
- **Critic bias.** The critic prompt is deliberately permissive ("only reject if 100 % normal structure"),
  which favours recall but may raise false positives.
- **Parse fallbacks default to good.** `SCREENING_FALLBACK_CLEAN` cells are recorded as `0` and should be
  reviewed manually.
- **Cost.** One screening call plus one critic call per candidate, so more calls than a single-pass approach.
- **Preprocessing.** The script does no contrast normalization. The images are already `_normalized`
  upstream and are converted from 8-bit grayscale to RGB without rescaling.
- The "lost in the middle" benefit is an expected effect of the design, not yet measured. Comparing against a
  single-pass baseline on labelled cells would confirm it.

## 5. Configuration

| Variable | Meaning | Default |
|----------|---------|---------|
| `OPENAI_BASE_URL` | vLLM/OpenAI-compatible endpoint | `http://localhost:8000/v1` |
| `MODEL_NAME` | fallback model id (server's first model is preferred) | `Qwen/Qwen3-VL-32B-Instruct` |
| `PANEL` | panel folder to process | `24-P10-A` |
| `LABEL_FONT_SIZE` | label font size on saved boxes | `28` |

## 6. Reference

Liu, N. F., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* arXiv:2307.03172.
