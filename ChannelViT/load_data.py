import torch
import re
import pandas as pd
from PIL import Image
import numpy as np
import sys
import os
import importlib
from torchvision import transforms
from utils import label_names

def save_image_from_array(image_array, file_path):
    # Ensure the array is in the correct format (uint8)
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    # Create an image from the array
    image = Image.fromarray(image_array)
    
    # Save the image
    image.save(file_path)
def stack_images(images_list):
    stacked_images = torch.cat([torch.tensor(image) for image in images_list], dim=0)
    return stacked_images

class Load_Data:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_pickle(self.path)
        self.df = self.df[self.df.labels >= 0]
        self.labels = self.df.labels.unique()
        self.labels_as_integers = [int(label) for label in self.df['labels'].values]
        self.images = [image.astype(np.float32) for image in list(self.df.images.values)]
        for image in self.images:
            if np.any(image > 255) or np.any(image < 0):
                raise ValueError("Image values should be in the range [0, 255]")
        self.images = [image.astype(np.uint8) for image in self.images]
        self.images = [np.array(Image.fromarray(image).convert('L')) for image in self.images]
        self.data = list(zip(self.images, self.labels_as_integers))

    def get_data(self):
        return self.data
    def get_just_images(self):
        return self.images
    def get_label_statistics(self):
        label_counts = {}
        for label in self.labels_as_integers:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        label_counts = dict(sorted(label_counts.items()))
        print(label_counts)
        label_counts_named = {(label_names())[int_label]: count for int_label, count in label_counts.items()}
        print(label_counts_named)
        return label_counts
    
    
class Load_Data_Handler:
    def __init__(self, PATH):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(current_dir, '../eagle-jsonhandler'))
        sys.path.append(os.path.join(current_dir, '../'))
        eagle_jsonhandler = importlib.import_module("JSONHandler")
        
        panellist = eagle_jsonhandler.getGroupedPanelList(PATH+'/overviews/')
        classified_cells = eagle_jsonhandler.getCellsByAttribute(groupedPanels=panellist, attribute="classifiedBy", values="Ralf")
        elpaths, uvpaths, vispaths, labels = eagle_jsonhandler.getCellsImagePathsAndLabels(classified_cells)
        label_types = ["good", "crack", "cross", "dark", "corrosion"]
        label_counts = np.sum(labels, axis=0)

        self.images_el = [np.array(Image.open(PATH+'/segments/'+path).convert('L')).astype(np.uint8) for path in elpaths]
        self.images_uv = [np.array(Image.open(PATH+'/segments/'+path).convert('L')).astype(np.uint8) for path in uvpaths]
        self.images_vis = [np.array(Image.open(PATH+'/segments/'+path).convert('L')).astype(np.uint8) for path in vispaths]
        
        # Combine the 3 lists of grayscale images into one list of RGB images
        # self.images = [np.stack((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        # self.images = [np.concatenate((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.images = [np.stack((el, uv, vis), axis=-1)for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.labels_as_integers = [np.where(label == 1)[0].tolist() for label in labels]
        self.data = list(zip(self.images, labels))

    def get_data(self):
        return self.data
    
    def get_label_statistics(self):
        label_counts = {}
        for label in self.labels_as_integers:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        return label_counts

class Load_Data_Handler_notlabeled:
    def __init__(self, PATH, subfolder):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        panelfolder = PATH+'/segments/'+subfolder
        elpaths = [file for file in os.listdir(panelfolder) if file.split("_")[1].startswith("EL") and file.endswith(".tif")]
        uvpaths = [file for file in os.listdir(panelfolder) if file.split("_")[1].startswith("UV") and file.endswith(".tif")]
        vispaths = [file for file in os.listdir(panelfolder) if file.split("_")[1].startswith("VI") and file.endswith(".tif")]
        label_types = ["good", "crack", "cross", "dark", "corrosion"]
        
        sorted_elpaths = sorted(elpaths, key=lambda x: (int(re.search(r'^(\d+)', x).group(1)), int(re.search(r'Cell(\d+)', x).group(1))))
        sorted_uvpaths = sorted(uvpaths, key=lambda x: (int(re.search(r'^(\d+)', x).group(1)), int(re.search(r'Cell(\d+)', x).group(1))))
        sorted_vispaths = sorted(vispaths, key=lambda x: (int(re.search(r'^(\d+)', x).group(1)), int(re.search(r'Cell(\d+)', x).group(1))))
        assert len(sorted_elpaths) == len(sorted_uvpaths) == len(sorted_vispaths), "Mismatch in number of images across channels"   

        self.images_el = [np.array(Image.open(PATH+'/segments/'+subfolder+'/'+path).convert('L')).astype(np.uint8) for path in sorted_elpaths]
        self.images_uv = [np.array(Image.open(PATH+'/segments/'+subfolder+'/'+path).convert('L')).astype(np.uint8) for path in sorted_uvpaths]
        self.images_vis = [np.array(Image.open(PATH+'/segments/'+subfolder+'/'+path).convert('L')).astype(np.uint8) for path in sorted_vispaths]

        # Combine the 3 lists of grayscale images into one list of RGB images
        # self.images = [np.stack((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        # self.images = [np.concatenate((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.images = [np.stack((el, uv, vis), axis=-1)for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.data = list(zip(self.images))
        self.sorted_elpaths=sorted_elpaths
        
    def get_data(self):
        return self.data, self.sorted_elpaths
    


class PVDataset(torch.utils.data.Dataset):
    def __init__(self, df, channels, transform=None, scale=1, return_labels=True):
        self.df = df
        self.channels = channels
        self.transform = transform
        self.scale = scale
        self.return_labels = return_labels

    def __getitem__(self, idx):
        row = self.df[idx]

        img_chw = Image.fromarray(row[0])

        # Apply data augmentation
        img_chw = self.transform(img_chw)

        # Select the specified channels
        if isinstance(img_chw, list):
            img_chw = [img[self.channels, :, :] for img in img_chw]
        else:
            img_chw = img_chw[self.channels, :, :]

        # Scale the channels if needed
        if self.scale != 1:
            if isinstance(img_chw, list):
                img_chw = [c * self.scale for c in img_chw]
            else:
                img_chw *= self.scale
        self.channels = torch.tensor([c for c in self.channels])

        if self.return_labels:
            return {"images": img_chw, "labels": row[1], "channels": self.channels}
        else:
            return {"images": img_chw, "channels": self.channels}
        
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)


def load_data_from_csv(csv_path):
    """
    Load data from a CSV file. Assumes the CSV contains 'image_path' and 'label' columns.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        list: A list of tuples where each tuple contains an image array and its corresponding label.
    """
    data = []
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        image_path = row['image_path']
        label = row['label']

        # Load the image and convert it to grayscale
        image = np.array(Image.open(image_path).convert('L')).astype(np.uint8)

        data.append((image, label))

    return data