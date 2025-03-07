import torch
import pandas as pd
from PIL import Image
import numpy as np
import sys
import os
import importlib
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
        self.images = [image.astype(np.uint8) for image in list(self.df.images.values)]
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
        print(label_counts)
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
        for i, count in enumerate(label_counts):
            print(f"Label {i}: {count} ({label_types[i]})")

        self.images_el = [np.array(Image.open(PATH+'/segments/'+path)) for path in elpaths]
        self.images_uv = [np.array(Image.open(PATH+'/segments/'+path)) for path in uvpaths]
        self.images_vis = [np.array(Image.open(PATH+'/segments/'+path)) for path in vispaths]
        
        # Combine the 3 lists of grayscale images into one list of RGB images
        # self.images = [np.stack((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        # self.images = [np.concatenate((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.images = [np.stack((el, uv, vis)) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.labels_as_integers = [np.argmax(label) for label in labels]
        self.data = list(zip(self.images, self.labels_as_integers))

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

class PVDataset(torch.utils.data.Dataset):
    def __init__(self, df, channels, transform=None, scale=1):
        self.df = df
        self.channels = channels
        self.transform = transform
        self.scale = scale

    def __getitem__(self, idx):
        row = self.df[idx]
        if row[0].ndim > 3:
            img_chw = [self.transform(Image.fromarray(channel)) for channel in row[0]]
            img_chw = stack_images(img_chw)
            # img_chw = [self.transform(img) if self.transform else img for img in img_chw_list]
        else:
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
        
        return {"images":img_chw, "labels": row[1], "channels": self.channels}

    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)
