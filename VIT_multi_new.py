import sys, os
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import torch
from transformers import ViTFeatureExtractor
from datasets import Dataset
from ChannelViT.VIT_utils import batch_transform_new, extractor, device, model_ckpt, compute_metrics, ContiguousDataCollator
from load_data import Load_Data
from utils import count_data_per_class, convert_labels_to_one_hot, count_data_per_multiclass


sys.path.insert(0, os.path.join("/Users/eagle/Documents/eagle-classification", "ChannelViT"))
################ LOAD DATA DURAMAT ################
num_classes = 5
path_Duramat = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
# directory_path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/"
data_loader =  Load_Data(path_Duramat)
data_Duramat = data_loader.get_data()
label_counts_duramat = count_data_per_class(data_Duramat)
count_data_per_multiclass(data_Duramat)
labels = [item[1] for item in data_Duramat]  # Extract the labels (tensors)
data_Duramat = convert_labels_to_one_hot(data_Duramat, num_classes=num_classes)
# remove DARK class because in Duramat it means something different
for i, item in enumerate(data_Duramat):
    item[1][3]=0
    data_Duramat[i] = (item[0], item[1]) 

################ LOAD DATA INFINITY ################


path_Infinity = "/Users/eagle/Documents//eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
data_loader =  Load_Data(path_Infinity)
data_Infinity_int = data_loader.get_data()
data_loader.get_label_statistics()
label_counts_infinity = count_data_per_class(data_Infinity_int)
count_data_per_multiclass(data_Infinity_int)

data_Infinity = convert_labels_to_one_hot(data_Infinity_int, num_classes)


################ ################################################

################ TRANSFORM DATA AND CREATE DATASETS ################

df_inf_transformed = batch_transform_new(data_Infinity)
ds_test_inf = Dataset.from_dict({
                "pixel_values": df_inf_transformed.pixel_values,
                "labels": df_inf_transformed.labels
            })
ds_test_inf

df_transformed_new = batch_transform_new(data_Duramat)
df_transformed_new.keys()
df_transformed_new['pixel_values'].shape, df_transformed_new['labels']


import numpy as np
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

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df_transformed_new.pixel_values, df_transformed_new.labels, test_size=.3, random_state=42, stratify=df_transformed_new.labels)
X_train.shape, X_val.shape
ds_val = Dataset.from_dict({
                "pixel_values": X_val,
                "labels": y_val
            })
ds_val

df_all = batch_transform_new(data_Duramat + data_Infinity)

ds_all = Dataset.from_dict({
                "pixel_values": df_all.pixel_values,
                "labels": df_all.labels
            })
trainer = Trainer(model=model,
                 args=training_args,
                 data_collator=data_collator,
                 compute_metrics=compute_metrics,
                 train_dataset=ds_all,
                 eval_dataset=ds_val,
                 tokenizer=extractor)

# train_result = trainer.train()

# import os, json

# save_dir = "/Users/eagle/Documents/eagle-classification/saved_vit_dur_inf_one_hot_5_classes_terminal/"
# os.makedirs(save_dir, exist_ok=True)

# # 1) save model and config (includes id2label/label2id)
# trainer.save_model(save_dir)            # saves model & config

# # 2) save feature extractor / image processor
# extractor.save_pretrained(save_dir)     # ViTFeatureExtractor / ViTImageProcessor

# # 3) save Trainer state (optional, useful for resuming)
# trainer.save_state()

# # 4) save any extra metadata (e.g. label order) explicitly
# with open(os.path.join(save_dir, "meta.json"), "w") as f:
#     json.dump({"id2label": model.config.id2label, "label2id": model.config.label2id}, f)


from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import Dataset
import torch, os, json
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


import numpy as np
from PIL import Image


def get_available_panels(base_dir):
    """Return panel names that contain an EL folder."""
    panels = []
    for root, _, _ in os.walk(base_dir):
        if os.path.basename(root).upper() == "EL":
            panels.append(os.path.basename(os.path.dirname(root)))
    return sorted(set(panels))


base_image_dir = "./normalized_images"

# Set panel selection here:
# - Use None to process all panels with EL folders.
# - Use a list like ["23-P09-D"] to process specific panel(s).
PANELS_TO_PROCESS = ["23-P09-D"]


def load_el_images(base_dir, panel_names=None):
    """Load images from EL folders, optionally filtered by panel names."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    loaded = []
    image_paths = []
    selected_set = set(panel_names) if panel_names is not None else None

    for root, _, files in os.walk(base_dir):
        if os.path.basename(root).upper() != "EL":
            continue

        panel_name = os.path.basename(os.path.dirname(root))
        if selected_set is not None and panel_name not in selected_set:
            continue

        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_ext:
                continue

            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img_arr = np.array(img, dtype=np.uint8)
                loaded.append((img_arr, torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)))
                image_paths.append(fpath)
            except Exception as e:
                print(f"Skipping unreadable image: {fpath} ({e})")

    print(f"Selected panels: {', '.join(panel_names) if panel_names is not None else 'ALL'}")
    print(f"Loaded {len(loaded)} images from EL folders under: {base_dir}")
    return loaded, image_paths


available_panels = get_available_panels(base_image_dir)
if not available_panels:
    raise ValueError(f"No EL folders found under: {base_image_dir}")

if PANELS_TO_PROCESS is None:
    selected_panels = None
else:
    unknown_panels = [p for p in PANELS_TO_PROCESS if p not in available_panels]
    if unknown_panels:
        raise ValueError(
            f"Unknown panel(s): {unknown_panels}. Available: {available_panels}"
        )
    selected_panels = PANELS_TO_PROCESS

df_test_converted, loaded_image_paths = load_el_images(base_image_dir, selected_panels)

if len(df_test_converted) == 0:
    raise ValueError(f"No EL images found under: {base_image_dir}")



df_transformed_new = batch_transform_new(df_test_converted)
ds_test_new = Dataset.from_dict({
                "pixel_values": df_transformed_new.pixel_values,
              "labels": df_transformed_new.labels,
            })

predictions_new = trainer_from_file.predict(ds_test_new)
predlabels_new = predictions_new.predictions.argmax(axis=-1)
#predictions_new.metrics
# ...existing code...
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


# plotting class 1 samples (replace previous cell)
import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()
# ...existing code...