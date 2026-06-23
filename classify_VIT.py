import os
import argparse
import torch
import numpy as np
import csv
from PIL import Image
from glob import glob
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from ChannelViT.VIT_utils import batch_transform_new
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--panel", default="23-P09-C", type=str, help="Panel name, e.g. 23-P09-C")
parser_args = parser.parse_args()

load_dir = "/Users/eagle/Documents/eagle-classification/saved_vit_dur_inf_one_hot_5_classes)terminal/"

#load_dir = "/Users/eagle/Documents/eagle-classification/Data/models/saved_vit_dur_inf_one_hot_5_classes/"
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


# load model + processor
model = ViTForImageClassification.from_pretrained(load_dir)
extractor = ViTFeatureExtractor.from_pretrained(load_dir)
model.to(device)

# optional: create minimal TrainingArguments & Trainer for predict
args = TrainingArguments(output_dir=load_dir, per_device_eval_batch_size=16, no_cuda=not torch.cuda.is_available())
trainer_from_file = Trainer(model=model, args=args, tokenizer=extractor)

base_image_dir = "./normalized_images"

# Set panel selection here:
# - Use None to process all panels with EL folders.
# - Use a list like ["23-P09-D"] to process specific panel(s).
panel = parser_args.panel
panel_dir = os.path.join(base_image_dir, panel)


def load_el_images(panel_dir):
    """Load images from EL folders in panel_dir"""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    loaded = []
    image_paths = []

    files = glob(os.path.join(panel_dir, "EL", "*"))

    for fpath in sorted(files):

        try:
            img = Image.open(fpath).convert("RGB")
            img_arr = np.array(img, dtype=np.uint8)
            loaded.append((img_arr, torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)))
            print(f"Loaded image: {fpath}")
            filename = os.path.basename(fpath)
            #print(f"Filename: {filename}")
            image_paths.append(filename)
        except Exception as e:
            print(f"Skipping unreadable image: {fpath} ({e})")

    print(f"Loaded {len(loaded)} images from EL folder under: {panel_dir}")
    return loaded, image_paths


df_test_converted, loaded_image_paths = load_el_images(panel_dir)

if len(df_test_converted) == 0:
    raise ValueError(f"No EL images found under: {base_image_dir}")


df_transformed_new = batch_transform_new(df_test_converted)
ds_test_new = Dataset.from_dict({
                "pixel_values": df_transformed_new.pixel_values,
              "labels": df_transformed_new.labels,
            })

predictions_new = trainer_from_file.predict(ds_test_new)
logits = predictions_new.predictions                      # shape (N, 5)
probs = 1 / (1 + np.exp(-logits))                        # sigmoid
pred_multilabel = (probs > 0.5).astype(int)              # shape (N, 5), multi-hot

# Example: get string labels per sample
LABEL_MAP = ['good', 'crack', 'cross', 'dark', 'corrosion']
INT_TO_LABEL = {i: l for i, l in enumerate(LABEL_MAP)}

label_strings = [
    ' '.join(INT_TO_LABEL[i] for i, v in enumerate(row) if v == 1) or 'good'
    for row in pred_multilabel
]
#predlabels_new = predictions_new.predictions.argmax(axis=-1)
#predictions_new.metrics


csv_path = f"OPENAI/{panel}/classification_results_VIT_{panel}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "classification"])
    for path, label in zip(loaded_image_paths, label_strings):
        print(f"Writing to CSV: {path},{label}")
        writer.writerow([path, label])

print(f"Saved {len(loaded_image_paths)} rows to {csv_path}")
