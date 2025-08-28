import warnings
import torch
import os
import re
import argparse
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split

#### Local imports
from training_multi_one_input_type import retrain_resume_or_load_pretrained_second_stage,retrain_resume_or_load_pretrained, init_trainer, train_save_model
from load_data import PVDataset, get_train_transforms, get_val_transforms, just_transform, Load_Data_Handler_notlabeled
from utils import (convert_list_of_arrays_to_labels, calculate_class_accuracy_one_hot,
                    class_label_save, label_names, logits_to_classes)
from plots import (save_images_by_label, plot_multilabel_confusion_matrix,
                    plot_samples_from_all_labels_with_acc, ploting_training_results)
from training_multi_one_input_type import load_post_trained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=16, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training and evaluation.')
    parser.add_argument('--retrain', type=str, default='predict_only', choices=['retrain', 'resume', 'predict_only'], help='Choose training mode: retrain, resume, or predict_only.')
    parser.add_argument('--use_only_EL', action='store_true', help='Use only EL images.')
    parser.add_argument('--all_colors', action='store_true', default=True, help='Use all available channels.')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')     
    parser.add_argument('--init_weights_name', type=str, default='so2sat_channelvit_small_p8_with_hcs_hard_split_supervised', help='Name of the initial weights file.')
    
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
    config['current_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['device'] = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    config['output_model_folder'] = os.path.join(f'/Data/models/model_with_{args.init_weights_name}/epochs_{args.num_train_epochs}/')
    config['input_model_folder'] = os.path.join( '/Data/models/')
    config['images_folder'] = os.path.join(config['current_dir'], 'Data/images/')
    config['path_Website'] = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"

    if args.all_colors:
        config['name_flag'] = 'rgb'
    else:
        config['name_flag'] = 'gray'

    if args.use_only_EL:
        config['channels'] = [0]
    else:
        if args.all_colors:
            config['channels'] = [0, 1, 2, 3, 4, 5, 6]
        else:
            config['channels'] = [0, 1, 2]

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


def load_all_data(config, args):
    print("Loading data...")
    filtered_data_path = os.path.join(config['images_folder'], "filtered_data.pkl")
    if os.path.exists(filtered_data_path):
        with open(filtered_data_path, "rb") as f:
            filtered_data = pickle.load(f)
        print(f"Filtered data loaded from {filtered_data_path}")
    else:
        raise FileNotFoundError(f"Filtered data not found at {filtered_data_path}. Please generate it first.")

    tensor_label_list_Duramat = just_transform(filtered_data['data_Duramat_filtered_more'], channels=[0])
    tensor_label_list_Infinity = just_transform(filtered_data['data_Infinity_filtered_more'], channels=[0], name='infinity')
    tensor_label_list_Website = just_transform(filtered_data['data_Website_filtered'] + filtered_data['data_Website_Ralf_filtered'], channels=config['channels'])

    cleaned_1channel_data = tensor_label_list_Duramat + tensor_label_list_Infinity
    cleaned_7channel_data = tensor_label_list_Website

    train_1channel_data, val_1channel_data = train_test_split(cleaned_1channel_data, test_size=0.2, random_state=42)
    train_7channel_data, val_7channel_data = train_test_split(cleaned_7channel_data, test_size=0.3, random_state=42)
    
    print(f"Loaded {len(cleaned_1channel_data)} one-channel samples and {len(cleaned_7channel_data)} seven-channel samples.")
    print(f"Splits: 1-channel (Train: {len(train_1channel_data)}, Val: {len(val_1channel_data)}), 7-channel (Train: {len(train_7channel_data)}, Val: {len(val_7channel_data)})")

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    dataset_train_1channel = PVDataset(train_1channel_data, channels=[[0]] * len(train_1channel_data), scale=1, return_labels=True, transform=train_transforms)
    dataset_val_1channel = PVDataset(val_1channel_data, channels=[[0]] * len(val_1channel_data), scale=1, return_labels=True, transform=val_transforms)
    dataset_train_7channel = PVDataset(train_7channel_data, channels=[config['channels']] * len(train_7channel_data), scale=1, return_labels=True, transform=train_transforms)
    dataset_val_7channel = PVDataset(val_7channel_data, channels=[config['channels']] * len(val_7channel_data), scale=1, return_labels=True, transform=val_transforms)

    return dataset_train_1channel, dataset_val_1channel, dataset_train_7channel, dataset_val_7channel


def run_predictions(trainer, dataset, dataset_name, out_folder, args, threshold=0.5):
    print(f"\n################# Running predictions for {dataset_name} #################")
    if not dataset:
        print(f"Skipping prediction for {dataset_name}: dataset is empty.")
        return

    predictions = trainer.predict(dataset) 
    pred_labels = logits_to_classes(predictions, initial_threshold=threshold)
    predlabels = convert_list_of_arrays_to_labels(pred_labels)

    empty_indices = [i for i, x in enumerate(predlabels) if not x]
    print(f"Number of samples with no predicted labels: {len(empty_indices)}")
    predlabels = [x for i, x in enumerate(predlabels) if i not in empty_indices]
    pred_labels = [x for i, x in enumerate(pred_labels) if i not in empty_indices]
    # Create a new dataset without the empty prediction samples for accurate metric calculation
    dataset_filtered = [item for i, item in enumerate(dataset) if i not in empty_indices]
    true_labels = np.array([item['labels'] for item in dataset_filtered])

    class_names = label_names()
    
    if len(true_labels) > 0:
        plot_multilabel_confusion_matrix(true_labels, pred_labels, class_names, output_path=os.path.join(out_folder, f'{dataset_name}_confusion_matrix.png'))
        
        print('################# ACCURACIES #################')
        class_accuracies = {}
        for label in range(args.num_classes):
            label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels, pred_labels, class_label=label)
            print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")

        plot_samples_from_all_labels_with_acc(dataset_filtered, predlabels, class_accuracies, data_name=dataset_name, 
                                            outfolder=out_folder, certainty=torch.sigmoid(torch.tensor(predictions[0])).numpy())
    else:
        print(f"No samples left for evaluation after filtering. Skipping plots.")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    args = parse_args()
    config = setup_config(args)

    # Use the appropriate device
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Correctly assign max_channels to the args object
    args.max_channels = config['max_channels']
    # First Stage: 1-channel data training
    dataset_train_1channel, dataset_val_1channel, dataset_train_7channel, dataset_val_7channel = load_all_data(config, args)

    print("\n----------------- STAGE 1: TRAINING ON 1-CHANNEL DATA -----------------")
    trainer = retrain_resume_or_load_pretrained(
        args, 
        config['current_dir'], 
        config['input_model_folder'], 
        config['device'], 
        config['output_model_folder'], 
        concat_train=dataset_train_1channel, 
        concat_val=dataset_val_1channel,
        channels=config['channels'], 
        name_flag=config['name_flag'] + 'Duramat'
    )
    
    run_predictions(trainer, dataset_val_1channel, '1-Channel_Validation', config['images_folder'], args)
    
    # Second Stage: 7-channel data fine-tuning
    print("\n----------------- STAGE 2: FINE-TUNING ON 7-CHANNEL DATA -----------------")
    args.retrain = 'retrain'
    trainer_7channel = retrain_resume_or_load_pretrained_second_stage(
        args, 
        config['current_dir'], 
        config['input_model_folder'], 
        config['device'], 
        config['output_model_folder'],
        concat_train=dataset_train_7channel, 
        concat_val=dataset_val_7channel
    )

    # Run predictions on the 7-channel validation set
    run_predictions(trainer_7channel, dataset_val_7channel, '7-Channel_Validation', config['images_folder'], args)
    
    # Final Prediction on unlabeled data
    print("\n----------------- PREDICTION ON UNLABELED DATA -----------------")
    
    # --- TISO DATA ---
    data_tiso_path = os.path.join(config['current_dir'], f'Data/processed_notlabeled_TISO_{config["name_flag"]}.pth')
    if os.path.exists(data_tiso_path):
        with open(data_tiso_path, 'rb') as f:
            data = torch.load(f)
            data_TISO_notlabeled = data['data_TISO_notlabeled_small']
    else:
        data_loader = Load_Data_Handler_notlabeled(config['path_Website'], args, 'TISO')
        data_TISO_notlabeled_raw, _ = data_loader.get_data()
        data_TISO_notlabeled = just_transform(data_TISO_notlabeled_raw, channels=config['channels'], notlabeled=True)
        with open(data_tiso_path, 'wb') as f:
            torch.save({'data_TISO_notlabeled_small': data_TISO_notlabeled}, f)

    dataset_tiso_notlabeled = PVDataset(data_TISO_notlabeled, channels=[config['channels']] * len(data_TISO_notlabeled), scale=1, return_labels=True)
    predictions_tiso = trainer_7channel.predict(dataset_tiso_notlabeled) 
    pred_labels_tiso = logits_to_classes(predictions_tiso, initial_threshold=0.9)
    predlabels_tiso = convert_list_of_arrays_to_labels(pred_labels_tiso)
    save_images_by_label(data_TISO_notlabeled, predlabels_tiso, os.path.join(config['images_folder'], 'data_TISO_notlabeled_good/'), flag='Website', name_flag=config['name_flag'])

    # --- C14 DATA ---
    data_c14_path = os.path.join(config['current_dir'], f'Data/processed_notlabeled_C14_{config["name_flag"]}.pth')
    if os.path.exists(data_c14_path):
        with open(data_c14_path, 'rb') as f:
            data = torch.load(f)
            data_C14_notlabeled = data['data_C14_notlabeled_small']
    else:
        data_loader = Load_Data_Handler_notlabeled(config['path_Website'], args, 'C14')
        data_C14_notlabeled_raw, _ = data_loader.get_data()
        data_C14_notlabeled = just_transform(data_C14_notlabeled_raw, channels=config['channels'], notlabeled=True)
        with open(data_c14_path, 'wb') as f:
            torch.save({'data_C14_notlabeled_small': data_C14_notlabeled}, f)

    dataset_c14_notlabeled = PVDataset(data_C14_notlabeled, channels=[config['channels']] * len(data_C14_notlabeled), scale=1, return_labels=True)
    predictions_c14 = trainer_7channel.predict(dataset_c14_notlabeled) 
    pred_labels_c14 = logits_to_classes(predictions_c14)
    predlabels_c14 = convert_list_of_arrays_to_labels(pred_labels_c14)
    save_images_by_label(data_C14_notlabeled, predlabels_c14, os.path.join(config['images_folder'], 'data_C14_notlabeled_good/'), flag='Website', name_flag=config['name_flag'])

    # --- INFINITY DATA ---
    data_infinity_path = os.path.join(config['current_dir'], f'Data/processed_notlabeledn_{config["name_flag"]}.pth')
    if os.path.exists(data_infinity_path):
        with open(data_infinity_path, 'rb') as f:
            data = torch.load(f)
            data_Infinity_notlabeled = data['data_Infinity_notlabeled_small']
    else:
        data_loader = Load_Data_Handler_notlabeled(config['path_Website'], args, 'Infinity')
        data_Infinity_notlabeled_raw, _ = data_loader.get_data()
        data_Infinity_notlabeled = just_transform(data_Infinity_notlabeled_raw, channels=config['channels'], notlabeled=True)
        with open(data_infinity_path, 'wb') as f:
            torch.save({'data_Infinity_notlabeled_small': data_Infinity_notlabeled}, f)

    dataset_infinity_notlabeled = PVDataset(data_Infinity_notlabeled, channels=[config['channels']] * len(data_Infinity_notlabeled), scale=1, return_labels=True)
    predictions_infinity = trainer_7channel.predict(dataset_infinity_notlabeled) 
    pred_labels_infinity = logits_to_classes(predictions_infinity)
    predlabels_infinity = convert_list_of_arrays_to_labels(pred_labels_infinity)
    save_images_by_label(data_Infinity_notlabeled, predlabels_infinity, os.path.join(config['images_folder'], 'data_Infinity_notlabeled_good/'), flag='Website', name_flag=config['name_flag'])


if __name__ == '__main__':
    main()