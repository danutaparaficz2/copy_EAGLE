import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import pandas as pd
PYTORCH_ENABLE_MPS_FALLBACK=1
from hubconf import camelyon_channelvit_small_p8_with_hcs_supervised, so2sat_channelvit_small_p8_with_hcs_random_split_supervised
from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small
#### Local imports
from load_data import Load_Data, PVDataset, Load_Data_Handler
from utils import compute_metrics, ploting_training_results
from transformers import TrainingArguments, Trainer
import matplotlib.pyplot as plt
import os

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs['labels']
        outputs = model(inputs['images'], extra_tokens=inputs)
        loss =  self.compute_loss_function(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_loss_function(self, outputs, labels):
        return torch.nn.functional.cross_entropy(outputs, labels)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        labels = inputs['labels']
        with torch.no_grad():
            outputs = model(inputs['images'], extra_tokens=inputs)
        if prediction_loss_only:
            loss = self.compute_loss_function(outputs, labels)
            return (loss, None, None)
        return (None, outputs, labels)
    
# we need to collate the data to be able to use multiple inputs images and labels and channels
def custom_collate_fn(examples):
    images = torch.stack([example['images'] for example in examples])
    rest = {k: default_collate([example[k] for example in examples]) for k in examples[0] if k != 'images'}
    return {'images': images, **rest}
    

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Usage

    path = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
    data_loader =  Load_Data(path)
    data = data_loader.get_data()
 

    # PATH_DATA = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    # data_loader_2 = Load_Data_Handler(PATH_DATA)
    # data = data_loader_2.get_data()

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_dataset = PVDataset(train_data, channels=[0, 1, 2], transform=transform, scale=1)
    val_dataset = PVDataset(val_data, channels=[0, 1, 2], transform=transform, scale=1)


    # this is an alternative way of loading the pretrained model (it doesn't work in Venus)
    # model = torch.hub.load('insitro/ChannelViT', 'imagenet_channelvit_small_p16_with_hcs_supervised', pretrained=True, map_location=torch.device('cpu'))

    # Load the model
    model = hcs_channelvit_small(patch_size=8, in_chans=18)
    # Load the pretrained weights and map them to the appropriate device
    state_dict = torch.load(current_dir+'/Data/so2sat_channelvit_small_p8_with_hcs_hard_split_supervised.pth', map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model.to(device)
    batch_size = 10
    logging_steps = len(train_data) // batch_size
    training_args = TrainingArguments(output_dir='./working/',
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    eval_strategy='epoch',
                                    save_strategy='epoch',
                                    num_train_epochs=15,
                                    fp16=True if torch.cuda.is_available() else False,
                                    logging_steps=logging_steps,
                                    learning_rate=1e-5,
                                    save_total_limit=2,
                                    remove_unused_columns=False,
                                    push_to_hub=False,
                                    metric_for_best_model='accuracy',
                                    load_best_model_at_end=True,
                                    logging_dir='./logs',  # Directory for storing logs
                                    ) 

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=custom_collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    train_result = trainer.train()
    # # Save the fine-tuned model
    torch.save(model.state_dict(), '/Data/finetuned_model_Duramat.pth')
    ploting_training_results(current_dir)

    # Load the fine-tuned model for evaluation
    model = hcs_channelvit_small(patch_size=8, in_chans=18)
    model.load_state_dict(torch.load(current_dir+'/Data/finetuned_model_Duramat.pth', map_location=device))
    model.eval()

    ############################## PREDICT ##############################
    predictions = trainer.predict(val_dataset) 
    predlabels = predictions.predictions.argmax(axis=-1)
    print(predictions.metrics)
    print('END Duramat')

    ########## INFINITY ##########
    path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
    data_loader =  Load_Data(path)
    data = data_loader.get_data()
    predictions = trainer.predict(data) 
    predlabels = predictions.predictions.argmax(axis=-1)
    print(predictions.metrics)
    print('END Duramat')