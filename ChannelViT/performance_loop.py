import sys
import argparse
import re
import os
import torch
from sklearn.model_selection import train_test_split
import numpy as _np


from load_data import PVDataset, get_train_transforms, get_val_transforms, Load_Data_Handler, count_data_per_class_in_labels
from utils import calculate_per_channel_stats, just_transform_with_norm, _ensure_len
from training_multi_one_input_type_stages import retrain_resume_or_load_pretrained_second_stage

############## PARSE ARGUMENTS AND SETUP CONFIGURATION ##############
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=18, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=13, help='Batch size for training and evaluation.')
    parser.add_argument('--retrain', type=str, default='retrain', choices=['retrain', 'resume', 'predict_only'], help='Choose training mode: retrain, resume, or predict_only.')
    parser.add_argument('--use_only_EL', action='store_true', default=False, help='Use only EL images.')
    parser.add_argument('--all_colors', action='store_true', default=True, help='Use all available channels.')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')     
    parser.add_argument('--init_weights_name', type=str, default='imagenet_channelvit_small_p16_with_hcs_supervised', help='Name of the initial weights file.')

    # In a notebook, avoid parsing the outer IPython argv — use an empty list.
    if ('ipykernel' in sys.modules) or ('get_ipython' in globals()):
        args = parser.parse_args([])   # safe for notebook
    else:
        args = parser.parse_args()

    def extract_number_from_name(name):
        match = re.search(r'p(\d+)', name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("No number found in the name")

    args.patch_size = extract_number_from_name(args.init_weights_name)
    return args


def setup_config(args):
    config = {}
    # __file__ is not defined inside notebooks — fall back to cwd when needed
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        base = os.getcwd()
    config['current_dir'] = base

    config['device'] = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    config['output_model_folder'] = os.path.join(f'/Data/models/model_with_{args.init_weights_name}/epochs_{args.num_train_epochs}/')
    config['input_model_folder'] = os.path.join('/Data/models/')
    config['images_folder'] = os.path.join(config['current_dir'], 'Data/images/')
    config['path_Website'] = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"

    config['name_flag'] = 'rgb' if args.all_colors else 'gray'
    if args.use_only_EL:
        config['channels'] = [0]
    else:
        config['channels'] = [0,1,2,3,4,5,6] if args.all_colors else [0,1,2]

    if args.init_weights_name == 'imagenet_channelvit_small_p16_with_hcs_supervised':
        config['max_channels'] = 3
    elif args.init_weights_name == 'so2sat_channelvit_small_p8_with_hcs_hard_split_supervised':
        config['max_channels'] = 18 
    elif args.init_weights_name == 'cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised':
        config['max_channels'] = 8 
    elif args.init_weights_name == 'camelyon_channelvit_small_p8_with_hcs_supervised':
        config['max_channels'] = 3 
    else:
        raise ValueError(f"Unknown init_weights_name: {args.init_weights_name}. Please set max_channels accordingly.")
    
    return config

args = parse_args()
config = setup_config(args)
print(config)
print(args)


train_transforms = get_train_transforms()
val_transforms = get_val_transforms()
###### ########## PREPARE DATALOADERS AND TRAIN IN THE LOOP EACH TECHNOLOGY SEPARATE ##############

predictions_all = []
label_ids_all = []
folder_names = []
for folder in ['23-P09-A', '23-P09-B', '23-P09-C', '23-P09-D', '23-P09-E', '23-P09-F', 
               '23-P09-G', '23-P09-H', 'C14-A', 'C14-B', 'C14-C', 'C14-D', 'C14-F', 'C14-G', 
               'C14-H', 'C14-I', 'C14-J', 'C14-K', 'Catamarano','PVVintage','TISO-EAGLE-23-P09_images', 
               'Infinity-alpin1','Infinity-moderate4']:
    ##### LOAD DATA FOR TRAINING AND VALIDATION #####
    print(f"\n================= Testing on folder: {folder} =================")
    path_Website = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    data_loader_2 = Load_Data_Handler(path_Website, args, classified_by=["Ebrar", "Ralf"], folders_excluded=[folder]) #, '23-P09-C', '23-P09-D', '23-P09-E', 'C14-A', 'C14-C','C14-I'
    data_Website = data_loader_2.get_data()
    label_counts_Website = count_data_per_class_in_labels(data_loader_2.labels_as_integers)
    total = sum(int(v) for _, v in label_counts_Website.items())
    print(f"Label counts for all except '{folder}' total={total}")

    calculated_mean, calculated_std = calculate_per_channel_stats(data_Website)
    tensor_label_list_Website = just_transform_with_norm(data_Website,
        calculated_mean=calculated_mean,
        calculated_std=calculated_std
    )
    print(f"Shape of first image tensor: {tensor_label_list_Website[0][0].shape}")
    print(f"Size of tensor_label_list_Website: {len(tensor_label_list_Website)}")
    ##### LOAD DATA FOR TESTING #####
    
    data_loader_test = Load_Data_Handler(path_Website, args, classified_by=["Ebrar", "Ralf"], this_folders_only=folder) #, '23-P09-C', '23-P09-D', '23-P09-E', 'C14-A', 'C14-C','C14-I'
    data_Website_test = data_loader_test.get_data()
    label_counts_Website_test = count_data_per_class_in_labels(data_loader_test.labels_as_integers)
    items = sorted(label_counts_Website_test.items())
    total = sum(int(v) for _, v in items)
    print(f"Label counts for '{folder}' total={total}: " + ", ".join(f"{k}:{int(v)}" for k, v in items))

    calculated_mean, calculated_std = calculate_per_channel_stats(data_Website_test)
    tensor_label_list_Website_test = just_transform_with_norm(data_Website_test,
        calculated_mean=calculated_mean,
        calculated_std=calculated_std
    )
    print(f"Shape of first image tensor: {tensor_label_list_Website_test[0][0].shape}")
    print(f"Size of tensor_label_list_Website: {len(tensor_label_list_Website_test)}")
    #  removes all-zero labels
    filtered = []
    removed = 0
    for pv, lbl in tensor_label_list_Website:
        ens = _ensure_len(lbl)          # normalize to length `num_classes`
        if not _np.all(ens == 0):      # drop all-zero labels
            filtered.append((pv, ens))
        else:
            removed += 1
    tensor_label_list_Website = filtered

    print(f"Filtered tensor_label_list_Website -> {len(tensor_label_list_Website)} examples (removed {removed} all-zero labels)")
    train_7channel_data, val_7channel_data = train_test_split(tensor_label_list_Website, test_size=0.3, random_state=42)

    dataset_train_7channel = PVDataset(train_7channel_data, channels=[config['channels']] * len(train_7channel_data), scale=1, return_labels=True, transform=train_transforms)
    dataset_val_7channel = PVDataset(val_7channel_data, channels=[config['channels']] * len(val_7channel_data), scale=1, return_labels=True, transform=val_transforms)
    dataset_Website_test = PVDataset(tensor_label_list_Website_test, channels=[config['channels']] * len(tensor_label_list_Website_test), scale=1, return_labels=True, transform=val_transforms)

    ##### TRAINING STAGE 2: FINE-TUNING ON 7-CHANNEL DATA #####
    print("\n----------------- STAGE 2: FINE-TUNING ON 7-CHANNEL DATA -----------------")
    args.retrain = 'retrain'
    args.batch_size = 10
    trainer_7channel = retrain_resume_or_load_pretrained_second_stage(
        args, 
        config['current_dir'], 
        config['input_model_folder'], 
        config['device'], 
        config['output_model_folder'],
        concat_train=dataset_train_7channel, 
        concat_val=dataset_val_7channel,
        folder=folder
    )

    ##### PREDICTION ON TEST SET #####
    print("\n----------------- PREDICTION ON TEST SET -----------------")

    predictions_web =trainer_7channel.predict(dataset_Website_test)
    predlabels_web = predictions_web.predictions.argmax(axis=-1)
    predictions_web.metrics

    predictions_all.append(predictions_web.predictions)
    label_ids_all.append(predictions_web.label_ids)
    folder_names.append(folder)

collect_predictions = {
    'folder': folder_names,
    'predictions': predictions_all,
    'label_ids': label_ids_all}
    # Save or process collect_predictions in file
save_path = os.path.join(config['current_dir'],  'predictions', f'predictions.pt')
torch.save(collect_predictions, save_path)
print(f"Predictions saved to {save_path}")