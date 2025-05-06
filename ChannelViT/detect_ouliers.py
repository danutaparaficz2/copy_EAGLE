import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import IsolationForest
from collections import defaultdict
import os
from shutil import copy2

if __name__ == '__main__':
        # Load pretrained model (feature extractor only)
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
    model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset (make sure it’s structured with subfolders for each class)
    dataset = ImageFolder('/Users/eagle/Documents/eagle-classification/Data/Duramat_ImageFolder/', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract features
    features = []
    labels = []
    paths = []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            feats = model(imgs).squeeze(-1).squeeze(-1)
            features.append(feats.numpy())
            labels.extend(lbls.numpy())
            batch_paths = [loader.dataset.samples[i][0] for i in range(len(labels) - len(lbls), len(labels))]
            paths.extend(batch_paths)

    features = np.vstack(features)
    labels = np.array(labels)


    outliers_per_class = defaultdict(list)

    for class_id in np.unique(labels):
        class_feats = features[labels == class_id]
        class_paths = np.array(paths)[labels == class_id]

        clf = IsolationForest(contamination=0.01)  # Tune contamination as needed
        preds = clf.fit_predict(class_feats)  # -1 for outliers

        for i, pred in enumerate(preds):
            if pred == -1:
                outliers_per_class[class_id].append(class_paths[i])

    output_dir = '/Users/eagle/Documents/eagle-classification/Outliers/'
    os.makedirs(output_dir, exist_ok=True)

    for cls, files in outliers_per_class.items():
        print(f"Class {cls} - {len(files)} outliers found:")
        class_out_dir = os.path.join(output_dir, f'class_{cls}')
        os.makedirs(class_out_dir, exist_ok=True)
        for f in files:
            print(f"  {f}")
            copy2(f, class_out_dir)



    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(features_scaled, labels)
    preds = clf.predict(features_scaled)
    probs = clf.predict_proba(features_scaled)

    # Flag mismatches or low-confidence samples
    mismatches = preds != labels
    low_conf = np.max(probs, axis=1) < 0.6  # adjust threshold

    suspect_idxs = np.where(mismatches | low_conf)[0]

    # Print suspect images
    print(f"\n⚠️ Found {len(suspect_idxs)} potentially mislabeled images:")
    for i in suspect_idxs:
        print(f"[Label: {labels[i]} | Predicted: {preds[i]} | Conf: {probs[i][preds[i]]:.2f}] → {paths[i]}")
        suspect_out_dir = os.path.join(output_dir, 'suspect_images')
        os.makedirs(suspect_out_dir, exist_ok=True)
        copy2(paths[i], suspect_out_dir)