
import os
# Prevent Transformers from importing TensorFlow / tf_keras which causes circular import in your venv
os.environ["TRANSFORMERS_NO_TF"] = "1"
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer


from plots import ploting_training_results
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import nn
from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small
import numpy as np
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
    images = torch.stack([example['images'].float() for example in examples])
    rest = {}
    for k in examples[0]:
        if k == 'images':
            continue
        items = []
        for example in examples:
            item = example[k]
            if isinstance(item, torch.Tensor):
                items.append(item.clone().detach().int())
            elif isinstance(item, np.ndarray):
                items.append(torch.tensor(item).int())
            elif isinstance(item, list):
                items.append(torch.tensor(item).int())
            else:
                items.append(torch.tensor(item).int())
        rest[k] = default_collate(items)
    return {'images': images, **rest}
    

def init_trainer(args, model, val_dataset, outfolder):

    training_args = TrainingArguments(output_dir=outfolder,
                                    per_device_train_batch_size= args.batch_size,
                                    per_device_eval_batch_size= args.batch_size,
                                    evaluation_strategy='epoch',
                                    save_strategy='epoch',
                                    num_train_epochs=args.num_train_epochs,
                                    fp16=False if torch.backends.mps.is_available() else False,
                                    logging_steps= args.batch_size,
                                    learning_rate=1e-5,
                                    save_total_limit=2,
                                    remove_unused_columns=False,
                                    push_to_hub=False,
                                    # metric_for_best_model='eval_accuracy',
                                    logging_dir='./logs',  # Directory for storing logs
                                    lr_scheduler_type="cosine", # Add this line
                                    warmup_ratio=0.1,
                                    gradient_accumulation_steps=5, # Experiment with this value
                                    load_best_model_at_end=True, # Add this
                                    metric_for_best_model="f1",  # Choose your metric
                                    greater_is_better=True,
                                    # dataloader_num_workers=os.cpu_count() # <-- Add this line
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

def train_save_model(trainer, train_dataset, val_dataset, outfolder, save=True):

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
    # ensure the output directory exists before saving
    if outfolder and not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
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
    model = hcs_channelvit_small(patch_size= args.patch_size, in_chans=4)
    # num_classes = 4  # Replace with the actual number of classes in your dataset
    model.head = nn.Linear(model.norm.normalized_shape[0], num_classes)

    model.to(device)

    # try to load pretrained weights and adapt channel embeddings if needed
    pretrained_file = os.path.join(folder, weights_path + '.pth')
    
    if os.path.exists(pretrained_file):
        print(f"Loading pretrained weights from {pretrained_file} and adapting channel embeddings...")
        state = torch.load(pretrained_file, map_location=device)
        state_dict = state.get('state_dict', state) if isinstance(state, dict) else state
        
        # Load ImageNet pretrained model for channels 1-3
        imagenet_file = '/Users/eagle/Documents/eagle-classification/Data/models/'+args.init_weights_name+'.pth'
        state1 = torch.load(imagenet_file, map_location=device)
        state_dict1 = state1.get('state_dict', state1) if isinstance(state1, dict) else state1
        
        emb_key = 'patch_embed.channel_embed.weight'
        if emb_key in state_dict and emb_key in state_dict1:
            old = state_dict[emb_key]      # Shape: (1, dim) - only channel 0 from Stage 1
            old1 = state_dict1[emb_key]    # Shape: (N, dim) - RGB/multi-channel from ImageNet
            
            print(f"\n=== Channel Embedding Initialization ===")
            print(f"Stage 1 channel embedding shape: {old.shape}")
            print(f"ImageNet channel embedding shape: {old1.shape}")
            
            if old.ndim == 2:
                old_ch, dim = old.shape
                new_ch = 4  # We need 4 channels now
                
                # Create new embedding tensor for 4 channels
                new_emb = torch.zeros((new_ch, dim), dtype=old.dtype, device=old.device)
                
                # Channel 0: Use embedding from Stage 1 (defect-trained, EL channel)
                new_emb[0] = old[0]
                print(f"✓ Channel 0: from Stage 1 (EL, defect-trained)")
                print(f"  Norm: {torch.norm(old[0]):.4f}")
                
                # Channels 1-3: Use embeddings from ImageNet (RGB channels)
                if old1.shape[0] >= 3:
                    new_emb[1:4] = old1[0:3]
                    print(f"✓ Channels 1-3: from ImageNet (RGB pretrained)")
                    for i in range(3):
                        print(f"  Channel {i+1} norm: {torch.norm(old1[i]):.4f}")
                else:
                    # Fallback: replicate channel 0 embedding
                    print(f"⚠ Warning: ImageNet model has only {old1.shape[0]} channels, replicating channel 0")
                    new_emb[1:4] = old[0].unsqueeze(0).repeat(3, 1)
                
                state_dict[emb_key] = new_emb
                
                print(f"\nChannel embedding initialization summary:")
                print(f"  Channel 0 norm: {torch.norm(new_emb[0]):.4f} (from Stage 1)")
                print(f"  Channel 1 norm: {torch.norm(new_emb[1]):.4f} (from ImageNet)")
                print(f"  Channel 2 norm: {torch.norm(new_emb[2]):.4f} (from ImageNet)")
                print(f"  Channel 3 norm: {torch.norm(new_emb[3]):.4f} (from ImageNet)")
                
                # Verify embeddings are differentiated
                similarities = []
                for i in range(4):
                    for j in range(i+1, 4):
                        sim = torch.cosine_similarity(new_emb[i], new_emb[j], dim=0).item()
                        similarities.append(sim)
                        if i == 0:  # Only print channel 0 similarities
                            print(f"  Similarity Ch0-Ch{j}: {sim:.4f}")
                
                if max(similarities) > 0.99:
                    print("⚠ WARNING: Some channel embeddings are too similar (>0.99)!")
                else:
                    print("✓ Channel embeddings are properly differentiated")
                print("="*50)
            else:
                print(f"Unexpected embedding tensor shape for {emb_key}: {old.shape}")

        # Load state dict with non-strict (to accept new head and adapted embeddings)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("\nLoaded pretrained weights (non-strict).")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        
        # --- Adaptive Fine-Tuning Stage 2, Phase A (Warm-up) Setup ---
        print("\n--- Initializing Phase A: Warm-up (Head & Channel Embeddings Trainable) ---")
        
        # 1. Freeze the entire model body (Transformer blocks, CLS, Pos Embeddings, etc.)
        model.requires_grad_(False)
                
        # 2. Explicitly unfreeze the parameters for Phase A training:
        # The new classification head (crucial for the new task)
        model.head.requires_grad_(True)
        # The adapted channel embeddings (crucial for the 4-channel input)
        model.patch_embed.channel_embed.weight.requires_grad_(True)
        
        # 3. Print verification of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters (Head + Channel Embeddings): {trainable_params / 1e6:.2f}M")
        # ----------------------------------------------------------------------
        
    else:
        print(f"No pretrained file at {pretrained_file}, initializing model from scratch.")

    return model


def unfreeze_for_phase_b(model):
    """
    Helper function to transition from Phase A (Warm-up) to Phase B (Full Fine-tuning).
    """
    print("\n--- Transitioning to Phase B: Full Fine-Tuning ---")
    
    # Unfreeze all parameters
    model.requires_grad_(True)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"All parameters are now trainable: {trainable_params / 1e6:.2f}M")
    
    # After calling this function, ensure your optimizer learning rate is set to a 
    # very low value (e.g., 1e-5) to prevent catastrophic forgetting.
    return model
 



def retrain_resume_or_load_pretrained_second_stage(args, current_dir, input_model_folder, device, output_model_folder,concat_train=None, concat_val=None, 
                                      channels=None, name_flag='', folder=''):
    # Model with originally pretrained weights
    if args.retrain == 'retrain':
        model = load_post_trained_model(args, current_dir+output_model_folder, device, 'trained_state_dict', args.num_classes)
        args.num_train_epochs = 5
        trainer = init_trainer(args, model, concat_val,  current_dir+output_model_folder)
        trainer = train_save_model(
            trainer,
            concat_train,   # train_dataset (should be your ConcatDataset)
            concat_val,     # val_dataset (should be your ConcatDataset)
            current_dir+output_model_folder+'/phase_b_7/'+folder+'/',
            save=False
        )
        # Unfreeze all layers for Phase B
        model = unfreeze_for_phase_b(model)
        args.num_train_epochs = 75

        trainer = init_trainer(args, model, concat_val,  current_dir+output_model_folder)
        trainer = train_save_model(
            trainer,
            concat_train,   # train_dataset (should be your ConcatDataset)
            concat_val,     # val_dataset (should be your ConcatDataset)
            current_dir+output_model_folder+'/phase_b_7/'+folder+'/',
            save=True
        )

        # ploting_training_results(trainer, current_dir+output_model_folder+'duramat_'+str(len(channels))+'channels'+name_flag+'/')

    else:
        model = load_post_trained_model(args, current_dir+output_model_folder, device, 'trained_state_dict', args.num_classes)
        # model = load_post_trained_model(args, current_dir+output_model_folder+'duramat_'+str(len(channels))+'channels'+name_flag+'/', device, 'trained_state_dict', args.num_classes)
        
        trainer = init_trainer(args, model, concat_val, current_dir+output_model_folder)

        checkpoints = [d for d in os.listdir(current_dir+output_model_folder+'/phase_b_7/'+folder) if d.startswith('checkpoint-')]
        if not checkpoints:
            print("No checkpoints found. Training the model.")
            trainer.train()
        else:
            print("Found checkpoints, loading the latest one.")
        # Sort checkpoints by the number in their name and get the latest one
        latest_checkpoint = os.path.join(current_dir+output_model_folder+'/phase_b_7/'+folder, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        # Restore state without continuing training
        trainer._load_from_checkpoint(latest_checkpoint)
        print(f"Trainer state restored from {latest_checkpoint}")
    return trainer


def retrain_resume_or_load_pretrained(args, current_dir, input_model_folder, device, output_model_folder,concat_train=None, concat_val=None, 
                                      channels=None, name_flag=''):
    # Model with originally pretrained weights
    if args.retrain == 'retrain':
        model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
        trainer = init_trainer(args, model, concat_val,  current_dir+output_model_folder)
        trainer = train_save_model(
            trainer,
            concat_train,   # train_dataset (should be your ConcatDataset)
            concat_val,     # val_dataset (should be your ConcatDataset)
            current_dir+output_model_folder
        )
        ploting_training_results(trainer, current_dir+output_model_folder)
    elif args.retrain == 'resume':
        model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
        args.num_train_epochs = 15
        trainer = init_trainer(args, model, concat_val, current_dir+output_model_folder)
        # Check if there are checkpoints in the output folder
        checkpoints = [d for d in os.listdir(current_dir+output_model_folder) if d.startswith('checkpoint-')]
        # Sort checkpoints by the number in their name and get the latest one
        latest_checkpoint = os.path.join(current_dir+output_model_folder, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
        # model = load_post_trained_model(args, current_dir+output_model_folder+'duramat_'+str(len(channels))+'channels'+name_flag+'/', device, 'trained_state_dict', args.num_classes)
        
        trainer = init_trainer(args, model, concat_val, current_dir+output_model_folder)
  
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
    return trainer
