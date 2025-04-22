from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from hubconf import camelyon_channelvit_small_p8_with_hcs_supervised, so2sat_channelvit_small_p8_with_hcs_random_split_supervised
import torch
from torch import nn

from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small

# My libraries
from load_data import  PVDataset
from utils import compute_metrics_sigmoid, augment_underrepresented_classes



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs['labels']
        outputs = model(inputs['images'], extra_tokens=inputs)
        loss_fct = nn.BCEWithLogitsLoss()
        # logits = outputs.logits

        loss = loss_fct(outputs, labels.float())
        return (loss, outputs) if return_outputs else loss

    def compute_loss_function(self, outputs, labels):
        return nn.BCEWithLogitsLoss(outputs, labels)
    
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
 #   images = {k: default_collate([example[k] for example in examples]) for k in examples[0] if k == 'images'}

    images = torch.stack([example['images'].float() for example in examples])
    rest = {k: default_collate([torch.tensor(example[k]).int() for example in examples]) for k in examples[0] if k != 'images'}
    return {'images': images, **rest}
    

def init_trainer(args, model, val_dataset, outfolder):

    training_args = TrainingArguments(output_dir=outfolder,
                                    per_device_train_batch_size= args.batch_size,
                                    per_device_eval_batch_size= args.batch_size,
                                    evaluation_strategy='epoch',
                                    save_strategy='epoch',
                                    num_train_epochs=args.num_train_epochs,
                                    fp16=True if torch.cuda.is_available() else False,
                                    logging_steps= args.batch_size,
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
        compute_metrics=compute_metrics_sigmoid,
        eval_dataset = val_dataset,
    )
    return trainer

def get_label_statistics(labels_as_integers):
    label_counts = {}
    for label in labels_as_integers:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    return label_counts
def data_just_transform(data, channels=[0]):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])
    dataset_all = PVDataset(data, channels=channels, transform=transform, scale=1)
    return  dataset_all

def data_split_and_transform(data, channels=[0, 1, 2]):
    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
    labels_as_integers = [item[1] for item in train_data]
    label_counts = get_label_statistics(labels_as_integers)

   # train_data = augment_underrepresented_classes(train_data, label_counts)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_dataset = PVDataset(train_data, channels=channels, transform=transform, scale=1)
    val_dataset = PVDataset(val_data, channels=channels, transform=transform, scale=1)
    return train_dataset, val_dataset, transform

def train_save_model(trainer, train_dataset, val_dataset, outfolder):

    trainer.train_dataset = train_dataset
    train_result = trainer.train()

    # print('!!!!!!!!!!!END TRAINING!!!!!!!!!!')
    predictions = trainer.predict(val_dataset) 
    print(predictions)
    print(predictions.metrics)
    print('END training')
    # # Save the trained model
    trainer.save_model(outfolder)  # This saves the model, tokenizer, and training arguments
    # Save the state of the trainer
    trainer.save_state()
    # Extract the state_dict from the trained model
    trained_state_dict = trainer.model.state_dict()
    torch.save(trained_state_dict, outfolder+'trained_state_dict.pth')
    return trainer

  # Load the model
def load_model(args, folder, device, weights_path):
    model = hcs_channelvit_small(patch_size= args.patch_size, in_chans=args.in_chans)
    model.load_state_dict(torch.load(folder+weights_path+'.pth', map_location=device))
    num_classes = 4  # Replace with the actual number of classes in your dataset
    model.head = nn.Linear(model.norm.normalized_shape[0], num_classes)
    model.to(device)
    return model

  # Load the model
def load_post_trained_model(args, folder, device, weights_path):
    model = hcs_channelvit_small(patch_size= args.patch_size, in_chans=args.in_chans)
    num_classes = 4  # Replace with the actual number of classes in your dataset
    model.head = nn.Linear(model.norm.normalized_shape[0], num_classes)
    model.load_state_dict(torch.load(folder+weights_path+'.pth', map_location=device))

    model.to(device)
    return model