from tkinter import Image
from turtle import pd
import warnings
import torch
import os
import re
import argparse
import numpy as np
import pickle
import json
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset
import time
import gc
#### Local imports
from training_multi_one_input_type_stages import retrain_resume_or_load_pretrained_second_stage,retrain_resume_or_load_pretrained, init_trainer, train_save_model
from load_data import PVDataset, get_train_transforms, get_val_transforms, just_transform, Load_Data_Handler_notlabeled, load_all_data_together
from utils import (convert_list_of_arrays_to_labels, calculate_class_accuracy_one_hot, convert_multilabels_to_one_hot, verify_data_normalization, check_raw_tensor_normalization, just_transform_with_norm,
                    class_label_save, label_names, logits_to_classes, logits_to_classes_TISO, just_transform_with_norm_without_label, check_component_normalization, calculate_raw_data_means, calculate_per_channel_stats)
from plots import (save_images_by_label, plot_multilabel_confusion_matrix,
                    plot_samples_from_all_labels_with_acc, ploting_training_results)
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=37, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=13, help='Batch size for training and evaluation.')
    #parser.add_argument('--retrain', type=str, default='predict_only', choices=['retrain', 'resume', 'predict_only'], help='Choose training mode: retrain, resume, or predict_only.')
    parser.add_argument('--retrain', type=str, default='retrain_second_stage', choices=['retrain', 'retrain_second_stage', 'resume', 'predict_only'],
                        help='Choose training mode: retrain, resume, or predict_only.')
    parser.add_argument('--use_only_EL', default=False, action='store_true', help='Use only EL images.')
    parser.add_argument('--all_colors', action='store_true', default=True, help='Use all available channels.')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')     
    parser.add_argument('--init_weights_name', type=str, default='imagenet_channelvit_small_p16_with_hcs_supervised', help='imagenet_channelvit_small_p16_with_hcs_supervised, cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised Name of the initial weights file.')
    parser.add_argument('--seed', type=int, default=15, help='Random seed for reproducibility.')
    #parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
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
            config['channels'] = [0, 1, 2 ,3]
        else:
            config['channels'] = [0]

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


def run_predictions(trainer, dataset, dataset_name, out_folder, args, threshold=0.5, fold=0):
    print(f"\n################# Running predictions for {dataset_name} #################")
    if not dataset:
        print(f"Skipping prediction for {dataset_name}: dataset is empty.")
        return

    num_labels=7
    import numpy as np
    label_names  =  {  0: 'good',
                1: 'crack',
                2: 'cross',
                3: 'dark',
                4: 'corrosion',
                5: 'discoloration',
                6: 'delamination',
            }

    # Evaluation cell - uses existing variables: predictions, num_labels, label_names (dict)
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                precision_recall_fscore_support, hamming_loss,
                                multilabel_confusion_matrix, accuracy_score, roc_auc_score)
    predictions = trainer.predict(dataset) 

    pred_labels, prob = logits_to_classes(predictions, initial_threshold=threshold)
    predlabels = convert_list_of_arrays_to_labels(pred_labels)

    empty_indices = [i for i, x in enumerate(predlabels) if not x]
    print(f"Number of samples with no predicted labels: {len(empty_indices)}")
    predlabels = [x for i, x in enumerate(predlabels) if i not in empty_indices]
    pred_labels = [x for i, x in enumerate(pred_labels) if i not in empty_indices]
    # Create a new dataset without the empty prediction samples for accurate metric calculation
    dataset_filtered = [item for i, item in enumerate(dataset) if i not in empty_indices]
    true_labels = np.array([item['labels'] for item in dataset_filtered])

    # prepare predictions and ground truth
    logits = predictions.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    true = predictions.label_ids.astype(int)

    class_accuracies = {}
    for label in range(args.num_classes):
            label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels, pred_labels, class_label=label)
            print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")

    plot_samples_from_all_labels_with_acc(dataset_filtered, predlabels, class_accuracies, data_name=dataset_name, 
                                            outfolder=out_folder, certainty=torch.sigmoid(torch.tensor(predictions[0])).numpy())
    # overall multi-label metrics
    metrics = {
        "f1_micro": f1_score(true, preds, average="micro"),
        "f1_macro": f1_score(true, preds, average="macro"),
        "precision_micro": precision_score(true, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(true, preds, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(true, preds),
        "subset_accuracy": float(np.mean(np.all(true == preds, axis=1))),
    }

    print("Overall metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # per-class precision/recall/f1/support
    prec, rec, f1, sup = precision_recall_fscore_support(true, preds, average=None, zero_division=0)
    print("\nPer-class metrics:")
    for i in range(num_labels):
        name = label_names.get(i, str(i)) if isinstance(label_names, dict) else str(i)
        print(f"  [{i}] {name:12s}  precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}  support={int(sup[i])}")

    # per-class confusion components (TN, FP, FN, TP)
    mlcm = multilabel_confusion_matrix(true, preds)
    print("\nPer-class confusion (TN, FP, FN, TP):")
    for i, cm in enumerate(mlcm):
        tn, fp, fn, tp = cm.ravel()
        name = label_names.get(i, str(i)) if isinstance(label_names, dict) else str(i)
        print(f"  [{i}] {name:12s}  TN={tn:5d}  FP={fp:5d}  FN={fn:5d}  TP={tp:5d}")

    # top-1 accuracy on examples that are single-label in ground truth

    # if len(true_labels) > 0:
    #     save_predictions(true_labels, pred_labels, prob, fold, dataset_name)
    #     '''
    #     class_names = label_names()
    #     plot_multilabel_confusion_matrix(true_labels, pred_labels, class_names, output_path=os.path.join(out_folder, f'{dataset_name}_confusion_matrix.png'))
        
    #     print('################# ACCURACIES #################')
    #     class_accuracies = {}
    #     for label in range(args.num_classes):
    #         label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels, pred_labels, class_label=label)
    #         print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")

    #     plot_samples_from_all_labels_with_acc(dataset_filtered, predlabels, class_accuracies, data_name=dataset_name, 
    #                                         outfolder=out_folder, certainty=torch.sigmoid(torch.tensor(predictions[0])).numpy())
    #     '''
    # else:
    #     print(f"No samples left for evaluation after filtering. Skipping plots.")


def save_predictions(true_labels, pred_labels, prob, fold, data_name):
    os.makedirs('results_test22', exist_ok=True)
    save_path = f'results_test22/{data_name}_{fold}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'pred_labels': pred_labels,
            'true_labels': true_labels,
            'probabilities': prob
        }, f)
    print(f"Saved predictions and true labels to {save_path}")


def prepare_datasets_once(config, args):
    """Load filtered_data, compute stats/transforms and build full PVDataset objects once."""
    print("Preparing datasets (one-time)...")
    import csv

    from load_data import Load_Data
    from utils import  convert_labels_to_one_hot, count_data_per_multiclass
    num_classes = 7
    path_Duramat = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
    # directory_path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/"
    data_loader =  Load_Data(path_Duramat)
    data_Duramat = data_loader.get_data()
    count_data_per_multiclass(data_Duramat)
    data_Duramat = convert_labels_to_one_hot(data_Duramat, num_classes=num_classes)

    path_Infinity = os.path.dirname(config['current_dir'])+"/eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
    data_loader =  Load_Data(path_Infinity)
    data_Infinity_int = data_loader.get_data()
    data_loader.get_label_statistics()
    count_data_per_multiclass(data_Infinity_int)

    data_Infinity = convert_labels_to_one_hot(data_Infinity_int, num_classes=num_classes)

    data_Website = []
    cpkl_website_test_path = "/Users/eagle/Documents/eagle-classification/OPENAI/23-P09-D/dataset_arrays_23-P09-D.pkl"
    cpkl_website_path = "/Users/eagle/Documents/eagle-classification/OPENAI/24-P10-A/dataset_arrays_24-P10-A.pkl"
    import pandas as pd
    try:
        df_Website = pd.read_pickle(cpkl_website_path)
    except ModuleNotFoundError as e:
        print(f"Error loading pickle: {e}")
        print("Attempting to load with compatibility mode...")
        # Alternative: use pickle directly with custom unpickler
        import pickle
        
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Redirect old numpy modules to new ones
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)
        
        with open(cpkl_website_path, 'rb') as f:
            df_Website = CustomUnpickler(f).load()
    
    data_Website = convert_multilabels_to_one_hot(df_Website, num_classes=num_classes)

    try:
        df_Website_test = pd.read_pickle(cpkl_website_test_path)
    except ModuleNotFoundError as e:
        print(f"Error loading pickle: {e}")
        print("Attempting to load with compatibility mode...")
        # Alternative: use pickle directly with custom unpickler
        import pickle
        
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Redirect old numpy modules to new ones
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)
        
        with open(cpkl_website_test_path, 'rb') as f:
            df_Website_test = CustomUnpickler(f).load()
    
    df_Website_test = convert_multilabels_to_one_hot(df_Website_test, num_classes=num_classes)
    # Data is already in (C, H, W) format with shape (4, H, W)
    # No permutation needed - channels remain as [0, 1, 2, 3]

    tensor_label_list_Duramat = just_transform(data_Duramat, channels=[0])
    tensor_label_list_Infinity = just_transform(data_Infinity, channels=[0], name='infinity')

    calculated_mean, calculated_std = calculate_per_channel_stats(data_Website)
    tensor_label_list_Website = just_transform_with_norm(data_Website, calculated_mean=calculated_mean, calculated_std=calculated_std)

    calculated_mean, calculated_std = calculate_per_channel_stats(df_Website_test)
    tensor_label_list_Website_test = just_transform_with_norm(df_Website_test, calculated_mean=calculated_mean, calculated_std=calculated_std)

    cleaned_1channel_data = tensor_label_list_Duramat + tensor_label_list_Infinity
    cleaned_7channel_data = tensor_label_list_Website
    # Check all tensor data normalization
    duramat_stats = check_raw_tensor_normalization(tensor_label_list_Duramat, "Duramat", check_all=True)
    infinity_stats = check_raw_tensor_normalization(tensor_label_list_Infinity, "Infinity", check_all=True)  
    website_stats = check_raw_tensor_normalization(tensor_label_list_Website, "Website", check_all=True)
    website_test_stats = check_raw_tensor_normalization(tensor_label_list_Website_test, "Website", check_all=True)

    print("\n" + "="*60)
    print("RAW DATA SUMMARY:")
    print("="*60)
    if duramat_stats:
        print(f"Duramat:  mean={duramat_stats['mean']:.4f}, std={duramat_stats['std']:.4f}, range=[{duramat_stats['min']:.2f}, {duramat_stats['max']:.2f}] - {duramat_stats['assessment']}")
    if infinity_stats:
        print(f"Infinity: mean={infinity_stats['mean']:.4f}, std={infinity_stats['std']:.4f}, range=[{infinity_stats['min']:.2f}, {infinity_stats['max']:.2f}] - {infinity_stats['assessment']}")
    if website_stats:
        print(f"Website:  mean={website_stats['mean']:.4f}, std={website_stats['std']:.4f}, range=[{website_stats['min']:.2f}, {website_stats['max']:.2f}] - {website_stats['assessment']}")
    if website_test_stats:
        print(f"Website:  mean={website_test_stats['mean']:.4f}, std={website_test_stats['std']:.4f}, range=[{website_test_stats['min']:.2f}, {website_test_stats['max']:.2f}] - {website_test_stats['assessment']}")
    print("="*80)

    # static split for 1-channel (done once)
    train_1channel_data, val_1channel_data = train_test_split(cleaned_1channel_data, test_size=0.2, random_state=args.seed)

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    # For 1-channel data, channels should be [0] for each sample (not [[0]])
    full_dataset_train_1ch = PVDataset(train_1channel_data, channels=[[0] for _ in range(len(train_1channel_data))], scale=1, return_labels=True, transform=train_transforms)
    full_dataset_val_1ch = PVDataset(val_1channel_data, channels=[[0] for _ in range(len(val_1channel_data))], scale=1, return_labels=True, transform=val_transforms)

    # For 4-channel data, channels should be [0, 1, 2, 3] for each sample
    # config['channels'] should be [0] but we actually have 4 channels now
    channels_4ch = [[0, 1, 2, 3] for _ in range(len(cleaned_7channel_data))]
    full_dataset_7ch = PVDataset(cleaned_7channel_data, channels=channels_4ch, scale=1, return_labels=True, transform=train_transforms)
    # create a validation-version that uses val transforms for evaluation
    full_dataset_7ch_eval = PVDataset(cleaned_7channel_data, channels=channels_4ch, scale=1, return_labels=True, transform=val_transforms)
    full_dataset_7ch_test = PVDataset(tensor_label_list_Website_test, channels=channels_4ch, scale=1, return_labels=True, transform=val_transforms)

    print(f"Prepared datasets: 1ch train={len(full_dataset_train_1ch)}, 1ch val={len(full_dataset_val_1ch)}, 7ch total={len(full_dataset_7ch)}")
    return full_dataset_train_1ch, full_dataset_val_1ch, full_dataset_7ch, full_dataset_7ch_eval, cleaned_7channel_data, full_dataset_7ch_test

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
    # shuffle=True is important to ensure randomness before splitting.
    K = 5

    full_train_1ch, full_val_1ch, full_7ch, full_7ch_eval, cleaned_7channel_data, full_7ch_test = prepare_datasets_once(config, args)

    N_total_samples = len(cleaned_7channel_data)
    dummy_data = np.arange(N_total_samples)
    kf = KFold(n_splits=K, shuffle=True, random_state=4)

    print(f"Starting {K}-Fold Cross-Validation...")
    # keep original requested retrain mode so we don't accidentally mutate it across folds
    orig_retrain = args.retrain

    # Keep 1-channel datasets static (already prepared)
    dataset_train_1channel = full_train_1ch
    dataset_val_1channel = full_val_1ch


    # ADD THESE LINES HERE
    if torch.backends.mps.is_available():
        print(f"Using device: {torch.device('mps')}")
    else:
        print(f"Using device: {torch.device('cpu')}")
    print("\n----------------- STAGE 1: TRAINING ON 1-CHANNEL DATA -----------------")
    args.retrain = orig_retrain
    retrain = args.retrain
    if retrain == 'retrain_second_stage':
        args.retrain = 'predict_only'  # Skip first stage if only second stage retraining is desired
    
    if args.retrain == 'retrain':
        if os.path.exists(config['output_model_folder']):
            shutil.rmtree(config['output_model_folder'])
            print(f"Removed existing output folder: {config['output_model_folder']}")
            
    trainer = retrain_resume_or_load_pretrained(
        args, 
        config['current_dir'], 
        config['input_model_folder'], 
        config['device'], 
        config['output_model_folder'], 
        concat_train=dataset_train_1channel, 
        concat_val=dataset_val_1channel,
        channels=[0], 
        name_flag=config['name_flag'] + 'Duramat'
    )
  #  run_predictions(trainer, dataset_val_1channel, '1-Channel_Validation', config['images_folder'], args, fold=fold)

    for fold, (train_index, test_index) in enumerate(kf.split(dummy_data)):
        start_time = time.perf_counter()
        print(f"\n--- Fold {fold+1}/{K} ---")

        # Lightweight Subset wrappers only (no recompute of transforms or stats)
        dataset_train_7channel = Subset(full_7ch, train_index)
        # for evaluation use eval transforms (full_7ch_eval) with same indices
        dataset_val_7channel = Subset(full_7ch_eval, test_index)

        print(f"Fold data sizes: 7ch train={len(dataset_train_7channel)}, 7ch val={len(dataset_val_7channel)}")


        #run_predictions(trainer, dataset_val_1channel, '1-Channel_Validation', config['images_folder'], args)
        run_predictions(trainer, full_7ch_eval, '7-Channel_Validation', config['images_folder'], args, fold=fold)
        run_predictions(trainer, full_7ch_eval, '7-Channel_Validation', config['images_folder'], args, fold=fold)    

        # exit()
        # Second Stage: 7-channel data fine-tuning
        print("\n----------------- STAGE 2: FINE-TUNING ON 7-CHANNEL DATA -----------------")
        if retrain == 'retrain_second_stage':
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
            channels=config['channels']
        )

        # Run predictions on the 7-channel validation set
        run_predictions(trainer_7channel, dataset_val_7channel, '7-Channel_Validation', config['images_folder'], args, fold=fold)
        run_predictions(trainer_7channel, dataset_val_1channel, '1-Channel_Validation', config['images_folder'], args, fold=fold)
        run_predictions(trainer_7channel, full_7ch_test, '7-Channel_TEST', config['images_folder'], args, fold=fold)
        exit()
        # cleanup to avoid memory growth
        try:
            # move model weights off device to free MPS memory immediately
            if 'trainer' in globals() and trainer is not None and hasattr(trainer, "model"):
                try:
                    trainer.model.to("cpu")
                except Exception:
                    pass
            if 'trainer_7channel' in globals() and trainer_7channel is not None and hasattr(trainer_7channel, "model"):
                try:
                    trainer_7channel.model.to("cpu")
                except Exception:
                    pass
        except Exception:
            pass

        # drop references
        trainer = None
        trainer_7channel = None

        # close any matplotlib figures
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass

        # remove logging handlers if those accumulate
        try:
            import logging as _logging
            root = _logging.getLogger()
            for h in list(root.handlers):
                try:
                    root.removeHandler(h)
                except Exception:
                    pass
        except Exception:
            pass

        # force garbage collection and clear device cache
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        # restore original args.retrain to ensure next fold starts from same requested mode
        args.retrain = orig_retrain
        elapsed = time.perf_counter() - start_time
        print(f"Fold {fold+1} finished in {elapsed:.1f}s")
'''   
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
    #predlabels_tiso = convert_list_of_arrays_to_labels(pred_labels_tiso)
    #save_images_by_label(data_TISO_notlabeled, predlabels_tiso, 
    #                     os.path.join(config['images_folder'], 'data_TISO_notlabeled_good_new2/'), 
    #                     flag='Website', name_flag=config['name_flag'])
    #exit()
    save_predictions(predictions_tiso, pred_labels_tiso, args.seed, 'tiso')

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
    pred_labels_c14 = logits_to_classes(predictions_c14)
    #predlabels_c14 = convert_list_of_arrays_to_labels(pred_labels_c14)
    #save_images_by_label(data_C14_notlabeled, predlabels_c14, 
    #                     os.path.join(config['images_folder'], 'data_C14_notlabeled_good/'), 
    #                     flag='Website', name_flag=config['name_flag'])

    save_predictions(predictions_c14, pred_labels_c14, args.seed, 'c14')

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
    pred_labels_infinity = logits_to_classes(predictions_infinity)
    #predlabels_infinity = convert_list_of_arrays_to_labels(pred_labels_infinity)
    #save_images_by_label(data_Infinity_notlabeled, predlabels_infinity, 
    #                     os.path.join(config['images_folder'], 'data_Infinity_notlabeled_good/'), 
    #                     flag='Website', name_flag=config['name_flag'])
    save_predictions(predictions_infinity, pred_labels_infinity, args.seed, 'infinity')
''' 


if __name__ == '__main__':
    main()