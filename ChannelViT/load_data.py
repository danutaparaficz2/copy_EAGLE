import torch
import pandas as pd
from PIL import Image
import numpy as np

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
    
class PVDataset(torch.utils.data.Dataset):
    def __init__(self, df, channels, transform=None, scale=1):
        self.df = df
        self.channels = channels
        self.transform = transform
        self.scale = scale

    def __getitem__(self, idx):
        row = self.df[idx]
        img_hwc = Image.fromarray(row[0])
        # Apply data augmentation
        img_chw = self.transform(img_hwc)

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