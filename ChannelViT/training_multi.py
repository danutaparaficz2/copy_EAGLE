import os
import json
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from hubconf import camelyon_channelvit_small_p8_with_hcs_supervised, so2sat_channelvit_small_p8_with_hcs_random_split_supervised
import torch
from PIL import Image
from torch import nn
import numpy as np
from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small
from matplotlib import pyplot as plt
# My libraries
from load_data import  PVDataset
from utils import compute_metrics_sigmoid


def find_wrongly_labeled_images(trainer, concat_val, threshold=0.7, save_dir="wrongly_labeled"):
    """
    Find images in validation set that are likely wrongly labeled
    
    Args:
        trainer: Trained Hugging Face trainer
        concat_val: Validation dataset
        threshold: Confidence threshold for predictions (0.7 = 70% confidence)
        save_dir: Directory to save wrongly labeled images
    
    Returns:
        wrong_indices: List of indices of likely wrongly labeled samples
    """
    import os
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Make predictions on validation set
    print("Making predictions on validation set...")
    predictions = trainer.predict(concat_val)
    pred_logits = predictions.predictions
    true_labels = predictions.label_ids
    
    # Convert logits to probabilities
    pred_probs = torch.sigmoid(torch.tensor(pred_logits))
    
    wrong_indices = []
    high_confidence_wrong = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (true_label, pred_prob) in enumerate(zip(true_labels, pred_probs)):
        # Convert one-hot true label to indices
        true_indices = np.where(true_label == 1)[0].tolist()
        
        # Get high-confidence predictions (> threshold)
        high_conf_pred_indices = torch.where(pred_prob > threshold)[0].tolist()
        
        # Check if high-confidence predictions don't match true labels
        true_set = set(true_indices)
        pred_set = set(high_conf_pred_indices)
        
        # If predictions are confident but different from true labels
        # Only flag as wrong if there is no overlap between predicted and true labels
        if pred_set and true_set and pred_set.isdisjoint(true_set):
            confidence_score = pred_prob.max().item()

            wrong_indices.append(i)
            high_confidence_wrong.append({
            'index': i,
            'true_labels': true_indices,
            'pred_labels': high_conf_pred_indices,
            'confidence': confidence_score,
            'all_probs': pred_prob.tolist()
            })
            
            # Save the image for manual inspection
            sample = concat_val[i]
            save_wrongly_labeled_image(sample, i, true_indices, high_conf_pred_indices, 
                         confidence_score, save_dir)
    
    # Save summary
    save_wrong_labels_summary(high_confidence_wrong, save_dir)
    
    print(f"Found {len(wrong_indices)} potentially wrongly labeled images")
    with open(save_dir+"wrong_indices.json", "w") as f:
           json.dump(wrong_indices, f, indent=2)
    print(f"Images saved to {save_dir}/")
    return 


def save_wrongly_labeled_image(sample, index, true_labels, pred_labels, confidence, save_dir):
    """Save image with true vs predicted labels"""
    image = sample['images']
    
    # Convert tensor to numpy and normalize for display
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3 :  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))
    if image.shape[2] == 7:
                    # Split the RGB image into its channels
                    red_channel = image[:, :, 0]
                    green_channel = image[:, :, 1:4]
                    blue_channel = image[:, :, 4:7]

                    # Normalize and enhance contrast for each channel
                    red_channel = (red_channel - np.min(red_channel)) / (np.max(red_channel) - np.min(red_channel) + 1e-8) * 255

                    green_channel = (255- green_channel) #* 255 # Brighten green channel
                    blue_channel = (255-blue_channel) #* 255  # Brighten blue channel

                    # # Clip values to ensure they remain valid
                    # green_channel = np.clip(green_channel, 0, 255)
                    # blue_channel = np.clip(blue_channel, 0, 255)

                    red_image = Image.fromarray(red_channel.astype(np.uint8))
                    green_image = Image.fromarray(green_channel.astype(np.uint8))
                    blue_image = Image.fromarray(blue_channel.astype(np.uint8))

                    # Concatenate the channels horizontally
                    combined_image = Image.new('RGB', (red_image.width * 3, red_image.height))
                    combined_image.paste(red_image, (0, 0))
                    combined_image.paste(green_image, (red_image.width, 0))
                    combined_image.paste(blue_image, (red_image.width * 2, 0))
                    # combined_image.save(os.path.join(save_dir, f'image_{index}.png'))
                    # Add title to the saved image
                    plt.figure(figsize=(red_image.width * 3 / 100, red_image.height / 100))
                    plt.imshow(np.asarray(combined_image))
                    plt.title(f"Index: {index}\nTrue: {true_labels}\nPred: {pred_labels}\nConf: {confidence:.3f}")
                    plt.axis('off')
                    plt.savefig(os.path.join(save_dir, f'image_{index}.png'), bbox_inches='tight', dpi=150)
                    plt.close()
    else:
        # Handle different image formats

        # Denormalize image: original normalization was mean=0, std=1
        image = (image * 1.0) + 0.0  # Undo normalization (redundant here, but explicit)
        # Scale to [0, 255] for visualization
        image = np.clip(image, -3, 3)  # Clip extreme values for display
        image = ((image + 3) / 6 * 255).astype(np.uint8)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        plt.imshow(image.squeeze(), cmap='gray' if image.shape[-1] == 1 else None)
        plt.title(f"Index: {index}\nTrue: {true_labels}\nPred: {pred_labels}\nConf: {confidence:.3f}")
        plt.axis('off')
        
        # Save image
        plt.savefig(f"{save_dir}/wrong_label_{index}.png", bbox_inches='tight', dpi=150)
        plt.close()

def save_wrong_labels_summary(wrong_data, save_dir):
    """Save summary of wrongly labeled data to JSON"""
    import json
    
    summary_path = os.path.join(save_dir, 'wrong_labels_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(wrong_data, f, indent=2)
    
    print(f"Summary saved to {summary_path}")



class CustomTrainer(Trainer):
    def __init__(self, *args, train_batch_sampler=None, eval_batch_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_batch_sampler = train_batch_sampler
        self.eval_batch_sampler = eval_batch_sampler

    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=1,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_sampler=self.eval_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=1,
        )
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
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        labels = inputs['labels'].to(device)
        with torch.no_grad():
            outputs = model(inputs['images'].to(device), extra_tokens={'channels':inputs['channels'].to(device)})
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
    

def init_trainer(args, model, outfolder, sampler_train=None, sampler_val=None, concat_train=None, concat_val=None):

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
        train_batch_sampler=sampler_train,
        eval_batch_sampler=sampler_val,    
        train_dataset=concat_train,
        eval_dataset=concat_val)
    return trainer

def get_label_statistics(labels_as_integers):
    label_counts = {}
    for label in labels_as_integers:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    return label_counts
# def data_just_transform(data, channels=[0]):

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5), std=(0.5)),
#     ])
#     dataset_all = PVDataset(data, channels=channels, transform=transform, scale=1)
#     return  dataset_all

# def data_split_and_transform(data, channels=[0, 1, 2]):
#     # Split data into training and validation sets
#     train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
#     labels_as_integers = [item[1] for item in train_data]
#     label_counts = get_label_statistics(labels_as_integers)

#    # train_data = augment_underrepresented_classes(train_data, label_counts)

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#     ])
#     train_dataset = PVDataset(train_data, channels=channels, transform=transform, scale=1)
#     val_dataset = PVDataset(val_data, channels=channels, transform=transform, scale=1)
#     return train_dataset, val_dataset, transform

def train_save_model(trainer, train_dataset, val_dataset, outfolder):

    trainer.train_dataset = train_dataset
    train_result = trainer.train()

    # print('!!!!!!!!!!!END TRAINING!!!!!!!!!!')
    # predictions = trainer.predict(val_dataset) 
    # print(predictions)
    # print(predictions.metrics)
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
def load_model(args, folder, device, weights_path, num_classes):
    model = hcs_channelvit_small(patch_size= args.patch_size, in_chans=args.max_channels)
    model.load_state_dict(torch.load(folder+weights_path+'.pth', map_location=device))
    # num_classes = 4  # Replace with the actual number of classes in your dataset
    model.head = nn.Linear(model.norm.normalized_shape[0], num_classes)
    model.to(device)
    return model

  # Load the model
def load_post_trained_model(args, folder, device, weights_path, num_classes):
    model = hcs_channelvit_small(patch_size= args.patch_size, in_chans=args.max_channels)
    model.head = nn.Linear(model.norm.normalized_shape[0], num_classes)
    model.load_state_dict(torch.load(folder+weights_path+'.pth', map_location=device))

    model.to(device)
    return model



def retrain_resume_or_load_pretrained(args, current_dir, input_model_folder, device, output_model_folder, sampler_train=None, 
                                      sampler_val=None, concat_train=None, concat_val=None, channels=None, name_flag=''):

        # Model with originally pretrained weights

    if args.retrain == 'retrain':
        model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
        if isinstance(concat_train, list):
            concat_train2 = concat_train[1]
            concat_val2 = concat_val[1]
            sampler_val2 = sampler_val[1]
            sampler_train2 = sampler_train[1]
            concat_train = concat_train[0]
            concat_val = concat_val[0]
            sampler_val = sampler_val[0]
            sampler_train = sampler_train[0]
        trainer = init_trainer(args, model, current_dir+output_model_folder, sampler_train=sampler_train, sampler_val=sampler_val, 
                               concat_train=concat_train, concat_val=concat_val)
        trainer = train_save_model(
            trainer,
            concat_train,   # train_dataset (should be your ConcatDataset)
            concat_val,     # val_dataset (should be your ConcatDataset)
            current_dir+output_model_folder+'all_'+str(len(channels))+'channels'+name_flag+'/'
        )

        if concat_train2:
            trainer = init_trainer(args, trainer.model, current_dir+output_model_folder, sampler_train=sampler_train2, sampler_val=sampler_val2, 
                               concat_train=concat_train2, concat_val=concat_val2)
            trainer = train_save_model(
                trainer,
                concat_train2,   # train_dataset (should be your ConcatDataset)
                concat_val2,     # val_dataset (should be your ConcatDataset)
                current_dir+output_model_folder+'all_'+str(len(channels))+'channels'+name_flag+'second_stage/'
            )


        # ploting_training_results(trainer, current_dir+output_model_folder+'duramat_'+str(len(channels))+'channels'+name_flag+'/')
    elif args.retrain == 'resume':
        model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
        args.num_train_epochs = 15
        trainer = init_trainer(args, model, current_dir+output_model_folder, sampler_train=sampler_train, sampler_val=sampler_val, 
                               concat_train=concat_train, concat_val=concat_val)
        # Check if there are checkpoints in the output folder
        checkpoints = [d for d in os.listdir(current_dir+output_model_folder) if d.startswith('checkpoint-')]
        # Sort checkpoints by the number in their name and get the latest one
        latest_checkpoint = os.path.join(current_dir+output_model_folder, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
        # model = load_post_trained_model(args, current_dir+output_model_folder+'duramat_'+str(len(channels))+'channels'+name_flag+'/', device, 'trained_state_dict', args.num_classes)
        
        trainer = init_trainer(
                args,
                model,
                current_dir+output_model_folder,  # outfolder
                sampler_train=sampler_train,      # pass your train sampler
                sampler_val=sampler_val,          # pass your val sampler
                concat_train=concat_train,        # pass your train dataset
                concat_val=concat_val             # pass your val dataset
        )    
        checkpoints = [d for d in os.listdir(current_dir+output_model_folder) if d.startswith('checkpoint-')]
        if not checkpoints:
            print("No checkpoints found. Training the model.")
            trainer.train()
        else:
            print("Found checkpoints, loading the latest one.")
        # Sort checkpoints by the number in their name and get the latest one
        latest_checkpoint = os.path.join(current_dir+output_model_folder, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        # Restore state without continuing training
        trainer._load_from_checkpoint(latest_checkpoint)
        print(f"Trainer state restored from {latest_checkpoint}")
    # else:
    #     from training_var import CustomTrainer, train_save_model, load_model, load_post_trained_model
    #     if args.retrain:
    #         model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
    #         trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_model_folder+'/multi/'+'/classes_var_newD/')
    #         trainer.train()
    #         trainer = train_save_model(trainer, current_dir+output_model_folder+'/multi/'+'classes_var_newD_long/')
    #         # # ploting_training_results(trainer, current_dir+output_model_folder+'duramat_3classes_var/')
    #     else:
    #         model = load_post_trained_model(args, current_dir+output_model_folder+'/multi/'+'/classes_var_newD_long/', device, 'trained_state_dict', args.num_classes)
    #         trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_model_folder+'/multi/'+'/classes_var_newD_long/')
    return trainer