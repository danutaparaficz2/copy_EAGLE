import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import torch
from transformers import ViTFeatureExtractor
from datasets import Dataset
from VIT_utils import batch_transform_new, extractor, device, plot_predicted_classes, create_grid_visualization, pv_to_display
from load_data import Load_Data
from utils import count_data_per_class, convert_labels_to_one_hot, count_data_per_multiclass, label_names
from sklearn.model_selection import train_test_split

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

data_Infinity = convert_labels_to_one_hot(data_Infinity_int, num_classes=num_classes)

################ train or load train ################


# trainer, model, extractor = trainVIT(labels, data_Duramat, data_Infinity)

load_dir = "/Users/eagle/Documents/eagle-classification/saved_vit_dur_inf_one_hot_5_classes)terminal/"
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


# load model + processor
model = ViTForImageClassification.from_pretrained(load_dir)
extractor = ViTFeatureExtractor.from_pretrained(load_dir)
model.to(device)

# optional: create minimal TrainingArguments & Trainer for predict
args = TrainingArguments(output_dir=load_dir, per_device_eval_batch_size=16, no_cuda=not torch.cuda.is_available())
trainer_from_file = Trainer(model=model, args=args, tokenizer=extractor)

############# LOAD UNLABELED DATA AND PREDICT #############
panel = "24-P10-A"
path = "/Users/eagle/Documents/eagle-classification/"
csv_path = path + "/OPENAI/" + panel + "/classification_results_EL_" + panel + "_integer.csv"
df_test = pd.read_csv(csv_path, header=0, dtype=str)

# Convert CSV paths to image arrays (ignore label column)
df_test_converted = []
filenames = []

for image_path in df_test["filename"].dropna():
    image_path = image_path.strip()
    if not image_path:
        continue
    img = np.array(Image.open(path+'/normalized_images/'+panel+'/EL/'+image_path))

    # Ensure (224, 224, 3)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_rgb = np.concatenate([img, img, img], axis=-1)
    else:
        img_rgb = img

    df_test_converted.append(
        (img_rgb.astype(np.uint8), torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32))
    )
    filenames.append(image_path)

df_transformed_new = batch_transform_new(df_test_converted)
ds_test_new = Dataset.from_dict({
                "pixel_values": df_transformed_new.pixel_values,
                "labels": df_transformed_new.labels,
                "filename": filenames,
            })

predictions_new = trainer_from_file.predict(ds_test_new)
predlabels_new = predictions_new.predictions.argmax(axis=-1)

# Categorize samples by prediction
categories = {i: [] for i in range(num_classes)}
all=[]
for idx, pred_label in enumerate(predlabels_new):
    sample = ds_test_new[idx]
    img = pv_to_display(sample["pixel_values"])
    if pred_label == 0 and (img.max() < 200 or img.mean() < 50 or (img.max() - img.mean()) > 100):
        categories[3].append({
            "pixel_values": img,
            "labels": 3,
            "idx": idx,
            "filename": sample["filename"]
        })
        all.append((sample["filename"], 3))
        
    elif pred_label!=0 and (img.max() < 200 or img.mean() < 50 or (img.max() - img.mean()) > 100):
        categories[3].append({
            "pixel_values": img,
            "labels": [3, pred_label],
            "idx": idx,
            "filename": sample["filename"]
        })  
        all.append((sample["filename"], '3,' + str(pred_label)))
    else:

        categories[pred_label].append({
        "pixel_values": img,
        "labels": pred_label,
        "idx": idx,
        "filename": sample["filename"]
        })
        all.append((sample["filename"], pred_label))

# plotting class 1 samples (replace previous cell)
from pathlib import Path

# Create grid visualizations
vis_dir = Path('/Users/eagle/Documents/eagle-classification/ChannelViT/data')
vis_dir.mkdir(exist_ok=True)
csv_output_path = vis_dir / "predicted_labels.csv"
print(f"\n✓ Creating visualizations in {vis_dir}/")
labels = label_names(flag='Website')

combined_labels = {}
for label_id, samples in categories.items():
    if len(samples) > 0:
        output_path = vis_dir / panel /f"{labels[label_id]}.png"
        output_path.parent.mkdir(exist_ok=True)
        images = [s["pixel_values"] for s in samples]
        file_names = [s["filename"] for s in samples]
        create_grid_visualization(
            images,
            labels[label_id],
            str(output_path),
        )
        print(f"  {label_id}: {len(images)} images")

import csv
el_csv = f'/Users/eagle/Documents/eagle-classification/OPENAI/{panel}/classification_results_EL_{panel}_integer_ViT.csv'
print(all)
with open(el_csv, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "label"])
    for filename, labels_list in sorted(all, key=lambda x: x[0].lower()):
        
        writer.writerow([filename, labels_list])
print(f"\n✓ Predicted labels saved to {el_csv}")