import warnings
import torch
import os
import re
import argparse
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#### Local imports
from training_multi_one_input_type import retrain_resume_or_load_pretrained_second_stage,retrain_resume_or_load_pretrained, init_trainer, train_save_model
from load_data import PVDataset, get_train_transforms, get_val_transforms, just_transform, Load_Data_Handler_notlabeled, load_all_data_together
from utils import (convert_list_of_arrays_to_labels, calculate_class_accuracy_one_hot, verify_data_normalization, check_raw_tensor_normalization, just_transform_with_norm,
                    class_label_save, label_names, logits_to_classes, logits_to_classes_TISO, just_transform_with_norm_without_label, check_component_normalization, calculate_raw_data_means, calculate_per_channel_stats)
from plots import (save_images_by_label, plot_multilabel_confusion_matrix,
                    plot_samples_from_all_labels_with_acc, ploting_training_results)
from training_multi_one_input_type import load_post_trained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=36, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=13, help='Batch size for training and evaluation.')
    parser.add_argument('--retrain', type=str, default='predict_only', choices=['retrain', 'resume', 'predict_only'], help='Choose training mode: retrain, resume, or predict_only.')
    parser.add_argument('--use_only_EL', action='store_true', default=False, help='Use only EL images.')
    parser.add_argument('--all_colors', action='store_true', default=True, help='Use all available channels.')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')     
    parser.add_argument('--init_weights_name', type=str, default='imagenet_channelvit_small_p16_with_hcs_supervised', help='Name of the initial weights file.')
    
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
        load_all_data_together( config['current_dir'],  config['images_folder'], name_flag='rgb', args=args)
        raise FileNotFoundError(f"Filtered data not found at {filtered_data_path}. Please generate it first.")
    # calculate_raw_data_means(filtered_data['data_Duramat_filtered_more'] , 'data_Duramat_filtered_more', sample_size=None)
    # calculate_raw_data_means(filtered_data['data_Website_Ralf_filtered'] , 'data_Website_Ralf_filtered', sample_size=None)
    # calculate_raw_data_means(filtered_data['data_Website_filtered'] , 'data_Website_filtered', sample_size=None)
    tensor_label_list_Duramat = just_transform(filtered_data['data_Duramat_filtered_more'], channels=[0])
    tensor_label_list_Infinity = just_transform(filtered_data['data_Infinity_filtered_more'], channels=[0], name='infinity')
    calculated_mean, calculated_std = calculate_per_channel_stats(filtered_data['data_Website_filtered'] + filtered_data['data_Website_Ralf_filtered'])
    tensor_label_list_Website = just_transform_with_norm(filtered_data['data_Website_filtered'] + filtered_data['data_Website_Ralf_filtered'], calculated_mean=calculated_mean, calculated_std=calculated_std)


    ######################
    cleaned_1channel_data = tensor_label_list_Duramat + tensor_label_list_Infinity
    cleaned_7channel_data = tensor_label_list_Website


    # ADD RAW DATA NORMALIZATION CHECKS HERE
    print("\n" + "="*80)
    print("CHECKING RAW TENSOR DATA NORMALIZATION (BEFORE TRANSFORMS)")
    print("="*80)
    
    # Check all raw tensor data
    duramat_stats = check_raw_tensor_normalization(tensor_label_list_Duramat, "Duramat", check_all=True)
    infinity_stats = check_raw_tensor_normalization(tensor_label_list_Infinity, "Infinity", check_all=True)  
    website_stats = check_raw_tensor_normalization(tensor_label_list_Website, "Website", check_all=True)
    
    print("\n" + "="*60)
    print("RAW DATA SUMMARY:")
    print("="*60)
    if duramat_stats:
        print(f"Duramat:  mean={duramat_stats['mean']:.4f}, std={duramat_stats['std']:.4f}, range=[{duramat_stats['min']:.2f}, {duramat_stats['max']:.2f}] - {duramat_stats['assessment']}")
    if infinity_stats:
        print(f"Infinity: mean={infinity_stats['mean']:.4f}, std={infinity_stats['std']:.4f}, range=[{infinity_stats['min']:.2f}, {infinity_stats['max']:.2f}] - {infinity_stats['assessment']}")
    if website_stats:
        print(f"Website:  mean={website_stats['mean']:.4f}, std={website_stats['std']:.4f}, range=[{website_stats['min']:.2f}, {website_stats['max']:.2f}] - {website_stats['assessment']}")
    print("="*80)

    train_1channel_data, val_1channel_data = train_test_split(cleaned_1channel_data, test_size=0.2, random_state=42)
    train_7channel_data, val_7channel_data = train_test_split(cleaned_7channel_data, test_size=0.1, random_state=42)
    
    print(f"Loaded {len(cleaned_1channel_data)} one-channel samples and {len(cleaned_7channel_data)} seven-channel samples.")
    print(f"Splits: 1-channel (Train: {len(train_1channel_data)}, Val: {len(val_1channel_data)}), 7-channel (Train: {len(train_7channel_data)}, Val: {len(val_7channel_data)})")

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()


    
    dataset_train_1channel = PVDataset(train_1channel_data, channels=[[0]] * len(train_1channel_data), scale=1, return_labels=True, transform=train_transforms)
    dataset_val_1channel = PVDataset(val_1channel_data, channels=[[0]] * len(val_1channel_data), scale=1, return_labels=True, transform=val_transforms)
    dataset_train_7channel = PVDataset(train_7channel_data, channels=[config['channels']] * len(train_7channel_data), scale=1, return_labels=True, transform=train_transforms)
    dataset_val_7channel = PVDataset(val_7channel_data, channels=[config['channels']] * len(val_7channel_data), scale=1, return_labels=True, transform=val_transforms)


    # ADD NORMALIZATION CHECKS HERE
    print("\n" + "="*80)
    print("CHECKING DATA NORMALIZATION")
    print("="*80)

    # Check all tensor data normalization
    duramat_stats = check_raw_tensor_normalization(tensor_label_list_Duramat, "Duramat", check_all=True)
    infinity_stats = check_raw_tensor_normalization(tensor_label_list_Infinity, "Infinity", check_all=True)  
    website_stats = check_raw_tensor_normalization(tensor_label_list_Website, "Website", check_all=True)
    # Check 1-channel training data
    print("\n--- 1-Channel Training Data ---")
    results_1ch_train = verify_data_normalization(dataset_train_1channel, sample_size=50, verbose=True)
    
    # Check 1-channel validation data
    print("\n--- 1-Channel Validation Data ---")
    results_1ch_val = verify_data_normalization(dataset_val_1channel, sample_size=30, verbose=True)
    
    # Check 7-channel training data
    print("\n--- 7-Channel Training Data ---")
    results_7ch_train = verify_data_normalization(dataset_train_7channel, sample_size=50, verbose=True)
    
    # Check 7-channel validation data
    print("\n--- 7-Channel Validation Data ---")
    results_7ch_val = verify_data_normalization(dataset_val_7channel, sample_size=130, verbose=True)
    
    # Stop execution if normalization is wrong
    all_results = [results_1ch_train, results_1ch_val, results_7ch_train, results_7ch_val]
    if not all(r['mean_ok'] and r['std_ok'] for r in all_results):
        print("\n⚠️  WARNING: Data normalization issues detected!")
        print("Please check your data preprocessing and transforms.")
        # Uncomment to stop execution:
        # return
    else:
        print("\n✅ All datasets passed normalization check!")
    
    print("="*80)
    return dataset_train_1channel, dataset_val_1channel, dataset_train_7channel, dataset_val_7channel


def run_predictions(trainer, dataset, dataset_name, out_folder, args, threshold=0.5):
    print(f"\n################# Running predictions for {dataset_name} #################")
    if not dataset:
        print(f"Skipping prediction for {dataset_name}: dataset is empty.")
        return

    predictions = trainer.predict(dataset) 
    pred_labels, _ = logits_to_classes(predictions, initial_threshold=threshold)
    predlabels = convert_list_of_arrays_to_labels(pred_labels)

    empty_indices = [i for i, x in enumerate(predlabels) if not x]
    print(f"Number of samples with no predicted labels: {len(empty_indices)}")
    predlabels = [x for i, x in enumerate(predlabels) if i not in empty_indices]
    pred_labels = [x for i, x in enumerate(pred_labels) if i not in empty_indices]
    # Create a new dataset without the empty prediction samples for accurate metric calculation
    dataset_filtered = [item for i, item in enumerate(dataset) if i not in empty_indices]
    true_labels = np.array([item['labels'] for item in dataset_filtered])

    class_names = label_names()

    plot_multilabel_confusion_matrix(true_labels, pred_labels, class_names, output_path=os.path.join(out_folder, f'Infinity_confusion_matrix.png'))

    if len(true_labels) > 0:
        for col in range(args.num_classes):
            col_preds = np.array(pred_labels)[:, col]
            col_trues = true_labels[:, col]
            print(f"  Column {col}:")
            print("    Accuracy:", accuracy_score(col_trues, col_preds))
        # plot_multilabel_confusion_matrix(true_labels, pred_labels, class_names, output_path=os.path.join(out_folder, f'{dataset_name}_confusion_matrix.png'))
        
        # print('################# ACCURACIES #################')
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


    # ADD THESE LINES HERE
    if torch.backends.mps.is_available():
        print(f"Using device: {torch.device('mps')}")
    else:
        print(f"Using device: {torch.device('cpu')}")
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
    
    # run_predictions(trainer, dataset_val_1channel, '1-Channel_Validation', config['images_folder'], args)

    # Second Stage: 7-channel data fine-tuning
    print("\n----------------- STAGE 2: FINE-TUNING ON 7-CHANNEL DATA -----------------")
  #  args.retrain = 'retrain'
    args.batch_size = 10
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
    # exit()
    # --- TISO DATA ---
    print("\n--- TISO DATA ---")
    data_tiso_path = os.path.join(config['current_dir'], f'Data/processed_notlabeled_TISO_{config["name_flag"]}.pth')
    if os.path.exists(data_tiso_path):
        with open(data_tiso_path, 'rb') as f:
            data = torch.load(f)
            data_TISO_notlabeled = data['data_TISO_notlabeled_small']
    else:
        data_loader = Load_Data_Handler_notlabeled(config['path_Website'], args, 'TISO')
        data_TISO_notlabeled_raw, _ = data_loader.get_data()
        calculated_mean, calculated_std = calculate_per_channel_stats(data_TISO_notlabeled_raw)
        data_TISO_notlabeled = just_transform_with_norm_without_label(data_TISO_notlabeled_raw, calculated_mean=calculated_mean, calculated_std=calculated_std)

        # data_TISO_notlabeled = just_transform(data_TISO_notlabeled_raw, channels=config['channels'], notlabeled=True)
        with open(data_tiso_path, 'wb') as f:
            torch.save({'data_TISO_notlabeled_small': data_TISO_notlabeled}, f)
            print(f"Saved TISO not labeled data to {data_tiso_path}")
            
    TISO_stats = check_raw_tensor_normalization(data_TISO_notlabeled, "Website", check_all=True)
    print(f"Website:  mean={TISO_stats['mean']:.4f}, std={TISO_stats['std']:.4f}, range=[{TISO_stats['min']:.2f}, {TISO_stats['max']:.2f}] - {TISO_stats['assessment']}")

    dataset_tiso_notlabeled = PVDataset(data_TISO_notlabeled, channels=[config['channels']] * len(data_TISO_notlabeled), 
                                        scale=1, return_labels=True)
    predictions_tiso = trainer_7channel.predict(dataset_tiso_notlabeled) 
    pred_labels_tiso = logits_to_classes_TISO(predictions_tiso, initial_threshold=0.9)
    predlabels_tiso = convert_list_of_arrays_to_labels(pred_labels_tiso)
    save_images_by_label(data_TISO_notlabeled, predlabels_tiso, 
                         os.path.join(config['images_folder'], 'data_TISO_notlabeled_good_new3/'), 
                         flag='Website', name_flag=config['name_flag'])
    exit()
    # --- C14 DATA ---
    print("\n--- C14 DATA ---")
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
    pred_labels_c14, _ = logits_to_classes(predictions_c14)
    predlabels_c14 = convert_list_of_arrays_to_labels(pred_labels_c14)
    save_images_by_label(data_C14_notlabeled, predlabels_c14, 
                         os.path.join(config['images_folder'], 'data_C14_notlabeled_good/'), 
                         flag='Website', name_flag=config['name_flag'])

    # --- INFINITY DATA ---
    print("\n--- INFINITY DATA ---")
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
    pred_labels_infinity, _ = logits_to_classes(predictions_infinity)
    predlabels_infinity = convert_list_of_arrays_to_labels(pred_labels_infinity)
    save_images_by_label(data_Infinity_notlabeled, predlabels_infinity, 
                         os.path.join(config['images_folder'], 'data_Infinity_notlabeled_good/'), 
                         flag='Website', name_flag=config['name_flag'])


if __name__ == '__main__':
    main()