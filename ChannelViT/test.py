import torch
from typing import List, Union
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
PYTORCH_ENABLE_MPS_FALLBACK=1
from hubconf import camelyon_channelvit_small_p8_with_hcs_supervised, so2sat_channelvit_small_p8_with_hcs_random_split_supervised

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
        return img_chw, {"labels": row[1], "channels": self.channels}

    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)

if __name__ == '__main__':
    # Usage
    path = "/Users/danuta.paraficz/PycharmProjects/eagle-classification/Data/Duramat_no_pool_labels.pkl"
    data_loader = Load_Data(path)
    data = data_loader.get_data()

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_dataset = PVDataset(train_data, channels=[0, 1, 2], transform=transform, scale=1)
    val_dataset = PVDataset(val_data, channels=[0, 1, 2], transform=transform, scale=1)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #device = 'mps'
    # Load the model
    model = so2sat_channelvit_small_p8_with_hcs_random_split_supervised(pretrained=False)

    # Load the pretrained weights and map them to the appropriate device
    state_dict = torch.load('Data/so2sat_channelvit_small_p8_with_hcs_hard_split_supervised.pth', map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model.to(device)

    # Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # data augmontation part? rotation etc!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Training loop
    num_epochs = 5
    model.train()
    print(f"Size of training data: {len(train_data)}")
   
    for epoch in range(num_epochs):
        k=0
        for images, metadata in train_dataloader:
            images, metadata = images.to(device), {k: v.to(device) for k, v in metadata.items()}

            # Forward pass
            outputs = model(images, extra_tokens=metadata)
            loss = criterion(outputs, metadata['labels'])
            k = k+1
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), 'finetuned_model'+str(epoch)+'.pth')

            print(epoch, k)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    print("Fine-tuning complete.")
    # Save the fine-tuned model
    torch.save(model.state_dict(), 'finetuned_model.pth')

    def compute_metrics(predicted, orgin_labels):
        labels = orgin_labels
        preds = predicted
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {
            'accuracy': acc,
            'f1': f1
        }

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, metadata in val_dataloader:
            images, metadata = images.to(device), {k: v.to(device) for k, v in metadata.items()}
            outputs = model(images, extra_tokens=metadata)
            _, predicted = torch.max(outputs.data, 1)
            total += metadata['labels'].size(0)
            correct += (predicted == metadata['labels']).sum().item()
            print(correct)

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    print('END')
