import torch
import os
import numpy as np
import torch.nn as nn
from itertools import zip_longest
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import TrainingArguments
from load_data import PVDataset
from utils import compute_metrics_sigmoid, augment_underrepresented_classes
from torch.utils.data.dataloader import default_collate
from matplotlib import pyplot as plt
from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small
import os
import cv2
import numpy as np
from tqdm import tqdm

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



def loss_plot(val_loss, outfolder, flag=''):
    # Plot the losses
    plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss')
    plt.savefig(outfolder+'/loss_plot'+flag+'.png')
    plt.close()

class CustomTrainer:
    def __init__(self, model, args, train_dataset, train_dataset_web, val_dataset, val_dataset_web, device, output_dir):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.train_dataset_web = train_dataset_web
        self.val_dataset = val_dataset
        self.val_dataset_web = val_dataset_web
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_loss = float('inf')
        self.early_stop_patience = 5
        self.early_stop_counter = 5
        self.output_dir = output_dir
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)


    def train(self):
        # train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: custom_collate_fn(x, self.device))
        # train_dataloader_web = DataLoader(self.train_dataset_web, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: custom_collate_fn(x, self.device))
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=lambda x: custom_collate_fn(x, self.device))
        val_dataloader_web = DataLoader(self.val_dataset_web, batch_size=self.args.batch_size, shuffle=False, collate_fn=lambda x: custom_collate_fn(x, self.device))
        val_loss_list = []
        val_loss_web_list = []
        for epoch in range(self.args.num_train_epochs):
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: custom_collate_fn(x, self.device))
            train_dataloader_web = DataLoader(self.train_dataset_web, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: custom_collate_fn(x, self.device))
            self.model.train()
            train_loss = 0.0
            for batch, batch_web in zip_longest(train_dataloader, train_dataloader_web, fillvalue=None):
                if batch is not None:
                    images = batch['images'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images, extra_tokens=batch)
                    loss = self.criterion(outputs, labels.float())
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                if batch_web is not None:
                    images_web = batch_web['images'].to(self.device)
                    labels_web = batch_web['labels'].to(self.device)
                    self.optimizer.zero_grad()
                    outputs_web = self.model(images_web, extra_tokens=batch_web)
                    loss_web = self.criterion(outputs_web, labels_web.float())
                    loss_web.backward()
                    self.optimizer.step()
                    train_loss += loss_web.item()

            train_loss /= (len(train_dataloader) + len(train_dataloader_web))
            val_loss = self.evaluate(val_dataloader)
            val_loss_list.append(val_loss)
            print(f"Epoch {epoch + 1}/{self.args.num_train_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            val_loss_web = self.evaluate(val_dataloader_web)
            val_loss_web_list.append(val_loss_web)

            print(f"Epoch {epoch + 1}/{self.args.num_train_epochs}, Train Loss: {train_loss:.4f}, Val Loss Web: {val_loss_web:.4f}")
            # Step the scheduler
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss+val_loss_web/10. < self.best_val_loss:
                self.best_val_loss = val_loss + val_loss_web/10.
                self.early_stop_counter = 0
                self.save_model(self.output_dir)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print("Early stopping triggered")
                    break
            # Compute metrics on the validation dataset
            val_metrics = self.compute_metrics(val_dataloader)
            print(f"Validation Metrics: {val_metrics}")
            val_web_metrics = self.compute_metrics(val_dataloader_web)
            print(f"Validation Web Metrics: {val_web_metrics}")
        loss_plot(val_loss_list, self.output_dir)
        loss_plot(val_loss_web_list, self.output_dir, flag='web')



    def compute_metrics(self, dataloader):
        self.model.eval()
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(images, extra_tokens=batch)
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        from transformers import EvalPrediction
        p = EvalPrediction(predictions=all_outputs, label_ids=all_labels)

        metrics = compute_metrics_sigmoid(p)
        return metrics
    
    def predict(self, inputs):
        self.model.eval()
        all_outputs = []
        val_dataloader = DataLoader(inputs, batch_size=self.args.batch_size, shuffle=False, collate_fn=lambda x: custom_collate_fn(x, self.device))

        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['images'].to(self.device)
                outputs = self.model(images, extra_tokens=batch)
                all_outputs.append(outputs)
  

        all_outputs = torch.cat(all_outputs, dim=0)
        return all_outputs
    

    def evaluate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(images, extra_tokens=batch)
                loss = self.criterion(outputs, labels.float())
                val_loss += loss.item()

        val_loss /= len(dataloader)
        return val_loss

    def save_model(self, output_dir):

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir + 'best_model.pth')

def custom_collate_fn(examples, device):
    images = torch.stack([example['images'].float().to(device) for example in examples])
    rest = {k: default_collate([torch.tensor(example[k]).int().to(device) for example in examples]) for k in examples[0] if k != 'images'}
    return {'images': images, **rest}

def data_just_transform(data, channels=[0], return_labels=True):
    # Define a custom transform to normalize the images to the range [0, 1]

    if len(channels) > 1:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.73], std=[0.17])  # Normalize the image

        ])
    dataset_all = PVDataset(data, channels=channels, transform=transform, scale=1, return_labels=return_labels)
    return  dataset_all

def train_save_model(trainer, outfolder):

    # Extract the state_dict from the trained model
    trained_state_dict = trainer.model.state_dict()
    torch.save(trained_state_dict, outfolder+'trained_state_dict.pth')
    return trainer

  # Load the model
def load_model(args, folder, device, weights_path, num_classes):
    model = hcs_channelvit_small(patch_size= args.patch_size, in_chans=args.in_chans)
    model.load_state_dict(torch.load(folder+weights_path+'.pth', map_location=device))
    model.head = nn.Linear(model.norm.normalized_shape[0], num_classes)
    model.to(device)
    return model

  # Load the model
def load_post_trained_model(args, folder, device, weights_path, num_classes):
    model = hcs_channelvit_small(patch_size= args.patch_size, in_chans=args.in_chans)
    model.head = nn.Linear(model.norm.normalized_shape[0], num_classes)
    model.load_state_dict(torch.load(folder+weights_path+'.pth', map_location=device))

    model.to(device)
    return model

