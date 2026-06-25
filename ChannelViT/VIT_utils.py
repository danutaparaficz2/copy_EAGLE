
from PIL import Image, ImageFont, ImageDraw
from sklearn.model_selection import train_test_split
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import DefaultDataCollator # You might use DataCollatorWithPadding if you have variable-length sequences, but DefaultDataCollator works for fixed-size images.
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Dict, Tuple, Optional, List

import math

model_ckpt = 'google/vit-base-patch16-224-in21k'
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
extractor = ViTFeatureExtractor.from_pretrained(model_ckpt)   #problem_type="multi_label_classification"  --> label torch.tensor([[1, 1, 0, 0]])

# ...existing code...
def batch_transform_new(examples):
    """
    Accepts:
      - a list of tuples (numpy_image_array, torch_label_tensor)  OR
      - a list of dicts with keys like 'image'/'images' and 'labels'  OR
      - a dict with keys 'images' and 'labels' (as before)
    Returns extractor output with 'pixel_values' (pt) and stacked 'labels' (torch.Tensor).
    Automatically converts grayscale (H,W) or (H,W,1) arrays to RGB.
    """
    import numpy as np
    import torch
    from PIL import Image

    # normalize examples -> imgs, labs lists
    if isinstance(examples, dict):
        imgs = list(examples.get("images", examples.get("image", examples.get("pixel_values", []))))
        labs = list(examples.get("labels", []))
    elif isinstance(examples, (list, tuple)):
        imgs = []
        labs = []
        for e in examples:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                imgs.append(e[0])
                labs.append(e[1])
            elif isinstance(e, dict):
                imgs.append(e.get("image") or e.get("images") or e.get("pixel_values"))
                labs.append(e.get("labels") or e.get("label"))
            else:
                raise ValueError("Unsupported example element. Expect (image, label) or dict.")
    else:
        raise ValueError("Unsupported examples type for batch_transform")

    # convert each image array/tensor to a PIL RGB image
    pil_imgs = []
    for im in imgs:
        # convert torch.Tensor -> numpy
        if isinstance(im, torch.Tensor):
            arr = im.detach().cpu().numpy()
        else:
            arr = np.asarray(im)

        # handle grayscale -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=2)
        # ensure uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        pil_imgs.append(Image.fromarray(arr))

    # run extractor (returns dict with 'pixel_values' as torch.Tensor)
    inputs = extractor(pil_imgs, return_tensors="pt")

    # prepare labels: stack if tensors, else convert to tensor
    if len(labs) == 0:
        labels_tensor = torch.tensor([])
    else:
        if isinstance(labs[0], torch.Tensor):
            labels_tensor = torch.stack([l.detach().cpu() for l in labs])
        else:
            labels_tensor = torch.tensor(labs)

    inputs["labels"] = labels_tensor
    return inputs
# ...existing code...


class ContiguousDataCollator(DefaultDataCollator):
    """
    Custom collator to ensure the 'pixel_values' tensor is contiguous.
    This fixes the RuntimeError on certain backends like MPS.
    """
    def __call__(self, features):
        # 1. Use the parent class's call method to assemble the batch
        batch = super().__call__(features)
        
        # 2. Check for the 'pixel_values' key (your input tensor)
        if 'pixel_values' in batch:
            # 🌟 Apply the .contiguous() fix
            batch['pixel_values'] = batch['pixel_values'].contiguous()
            batch['labels'] = batch['labels'].contiguous()
        # 3. Return the modified batch
        return batch
    
def compute_metrics(p):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits = p.predictions
    # some HF Trainer return (logits, hidden) tuples
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = p.label_ids

    # multiclass / scalar labels case
    if labels.ndim == 1:
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    # 2D labels: could be one-hot (single-label) or multi-hot (multi-label)
    if labels.ndim == 2:
        row_sums = labels.sum(axis=1)
        # one-hot encoded single-label (exactly one '1' per row)
        if np.all(row_sums == 1):
            true = labels.argmax(axis=1)
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(true, preds),
                "f1": f1_score(true, preds, average="weighted"),
            }

        # multi-hot multi-label case -> use sigmoid + threshold, use micro/sample F1
        # apply sigmoid safely
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        true = labels.astype(int)
        f1_micro = f1_score(true, preds, average="micro")
        # subset accuracy (exact match)
        subset_acc = np.mean(np.all(true == preds, axis=1))
        return {
            "subset_accuracy": float(subset_acc),
            "f1_micro": float(f1_micro),
        }

    # fallback
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
def compute_metrics(p):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    logits = p.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = p.label_ids  # shape (batch, num_labels), floats 0/1

    # apply sigmoid to logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    true = labels.astype(int)

    # multi-label metrics
    f1_micro = f1_score(true, preds, average='micro')
    f1_samples = f1_score(true, preds, average='samples')
    subset_acc = float(np.mean(np.all(true == preds, axis=1)))

    return {
        "f1_micro": float(f1_micro),
        "f1_samples": float(f1_samples),
        "subset_accuracy": subset_acc
    }

def pv_to_display(pv, extractor=None):
    """Convert pixel_values (torch/numpy, possibly normalized) -> HWC uint8 image for plt.imshow."""
    import numpy as np
    # convert torch tensor -> numpy
    if hasattr(pv, "numpy"):
        arr = pv.numpy()
    else:
        arr = np.asarray(pv)

    # handle batched (1, C, H, W)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # channel-first (C, H, W) -> H, W, C
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # If already uint8, just clip
    if arr.dtype == np.uint8:
        return np.clip(arr, 0, 255)

    # try to undo extractor normalization if available
    try:
        if extractor is not None and hasattr(extractor, "image_mean") and hasattr(extractor, "image_std"):
            mean = np.array(extractor.image_mean).reshape(1, 1, -1)
            std = np.array(extractor.image_std).reshape(1, 1, -1)
            # if input is channel-first originally, above transpose handled it
            arr = arr * std + mean
    except Exception:
        pass

    # common cases: values in [0,1] -> scale by 255
    amax = arr.max()
    amin = arr.min()
    if amax <= 1.1 and amin >= -0.1:
        arr = (arr - (0.0 if amin >= 0 else amin))  # shift small negatives if present
        arr = arr * 255.0
    else:
        # otherwise linearly rescale to 0-255
        if amax > amin:
            arr = (arr - amin) / (amax - amin) * 255.0
        else:
            arr = np.zeros_like(arr)

    return np.clip(arr, 0, 255).astype(np.uint8)

def trainVIT(labels, data_Duramat, data_Infinity):

    df_transformed_new = batch_transform_new(data_Duramat)
    df_transformed_new.keys()
    df_transformed_new['pixel_values'].shape, df_transformed_new['labels']


    X_train, X_val, y_train, y_val = train_test_split(df_transformed_new.pixel_values, df_transformed_new.labels, test_size=.3, random_state=42, stratify=df_transformed_new.labels)
    X_train.shape, X_val.shape
    ds_val = Dataset.from_dict({
                    "pixel_values": X_val,
                    "labels": y_val
                })

    df_all = batch_transform_new(data_Duramat + data_Infinity)

    ds_all = Dataset.from_dict({
                    "pixel_values": df_all.pixel_values,
                    "labels": df_all.labels
                })

    num_labels = 5


    print(f"num_labels={num_labels}, labels_min={min(labels)}, labels_max={max(labels)}")
    # sanity: ensure label values fit expected range if they are ints
    try:
        if all(isinstance(l, (int, np.integer)) for l in labels):
            assert max(labels) < num_labels, "label values exceed num_labels-1"
    except AssertionError as e:
        print("Label range check failed:", e)

    model = ViTForImageClassification.from_pretrained(
        model_ckpt,
        num_labels=num_labels,

    )
    # set problem type to match your label tensors:
    # - single_label_classification for scalar int labels
    # - multi_label_classification for multi-hot vectors
    model.config.problem_type = "multi_label_classification"

    model = model.to(device)

    batch_size = 16
    logging_steps = 100
    training_args = TrainingArguments(output_dir='./working1/',
                                     per_device_train_batch_size=batch_size,
                                     per_device_eval_batch_size=batch_size,
                                     evaluation_strategy='epoch',
                                     save_strategy='epoch',
                                     num_train_epochs=5,
                                     fp16=True if torch.cuda.is_available() else False,
                                     no_cuda=True,
                                     learning_rate=1e-5,
                                     save_total_limit=2,
                                     remove_unused_columns=False,
                                     push_to_hub=False,
                                     load_best_model_at_end=True)



    print("Defined custom data collator: ContiguousDataCollator.")
    data_collator = ContiguousDataCollator()
    # Explicitly set the device to CPU
    DEVICE = torch.device('cpu') 

    # Make sure your model is moved to this device
    model.to(DEVICE)
    # ...existing code...

    trainer = Trainer(model=model,
                     args=training_args,
                     data_collator=data_collator,
                     compute_metrics=compute_metrics,
                     train_dataset=ds_all,
                     eval_dataset=ds_val,
                     tokenizer=extractor)

    train_result = trainer.train()

    import os, json

    save_dir = "/Users/eagle/Documents/eagle-classification/saved_vit_dur_inf_one_hot_5_classes_terminal/"
    os.makedirs(save_dir, exist_ok=True)

    # 1) save model and config (includes id2label/label2id)
    trainer.save_model(save_dir)            # saves model & config

    # 2) save feature extractor / image processor
    extractor.save_pretrained(save_dir)     # ViTFeatureExtractor / ViTImageProcessor

    # 3) save Trainer state (optional, useful for resuming)
    trainer.save_state()

    # 4) save any extra metadata (e.g. label order) explicitly
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump({"id2label": model.config.id2label, "label2id": model.config.label2id}, f)
    return trainer, model, extractor


def plot_predicted_classes(predlabels_new, ds_test_new, extractor):
    class1_idx = np.where(np.array(predlabels_new) == 0)[0]
    n_display = min(36, len(class1_idx))

    if n_display == 0:
        print("No predicted class 1 samples found.")
    else:
        sel = np.random.choice(class1_idx, size=n_display, replace=False)
        fig, axes = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(20,20))
        axes = axes.flatten()
        for ax, idx in zip(axes, sel):
            s = ds_test_new[int(idx)]
            pv = s['pixel_values']
            img = pv_to_display(pv, extractor=extractor)  # extractor optional
            ax.imshow(img)
            ax.set_title(f"P: {int(predlabels_new[int(idx)])}")
            ax.axis('off')
        for ax in axes[n_display:]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("predicted_class_1_samples.png")



def create_grid_visualization(
    images: np.ndarray,
    category: str,
    output_path: str,
    thumb_size: int = 360,
    cols: int = 5
    ):
    """Create a grid visualization of images.

    If dark_output_path is provided, images that fail the brightness check
    are saved into a separate grid image at that path.
    """
    if not images:
        print(f"  No images for category: {category}")
        return
    
    def _render_grid(grid_images: List[Image.Image], title_text: str, path: str) -> None:
        n_items = len(grid_images)
        rows = math.ceil(n_items / cols)

        pad = 24
        title_pad = 80
        canvas_width = cols * thumb_size + (cols + 1) * pad
        canvas_height = rows * thumb_size + (rows + 1) * pad + title_pad
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
        except Exception:
            font = ImageFont.load_default()

        draw.text((pad, 10), title_text, fill="black", font=font)

        for idx, img in enumerate(grid_images):
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            row = idx // cols
            col = idx % cols
            x = col * thumb_size + (col + 1) * pad
            y = row * thumb_size + (row + 1) * pad + title_pad
            canvas.paste(img, (x, y))

        canvas.save(path, quality=95)
        print(f"  ✓ Saved visualization: {path}")


    # Place thumbnails
    main_images: List[Image.Image] = []

    for idx, img in enumerate(images):
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img_pil = Image.fromarray(img)
        else:
            img_pil = img

        main_images.append(img_pil)

    _render_grid(main_images, f"{category} ({len(main_images)} images)", output_path)
