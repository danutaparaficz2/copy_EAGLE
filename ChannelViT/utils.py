from sklearn.metrics import f1_score, accuracy_score, log_loss
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import torch
from transformers import  EvalPrediction
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
import random
import os
import json
from PIL import Image, ImageEnhance, ImageOps
import math
from torchvision.transforms import RandomAffine
from torchvision import transforms
from tqdm import tqdm



def select_images_by_label(ds, predlabels, label, certainty=None):
    selected_data = []
    selected_predlabels = []
    selected_certainty = []
    for idx, s in enumerate(ds):
        if s['labels'][label] == 1:
                selected_data.append(s)
                if predlabels is not None:
                    selected_predlabels.append(predlabels[idx])
                    selected_certainty.append(certainty[idx] if certainty is not None else None)
                else:
                    selected_predlabels = None
                    selected_certainty = None

    return selected_data, selected_predlabels, selected_certainty


def threshold_and_max(arr, threshold=0.5):
    result = (arr > threshold).astype(int)
    for i, row in enumerate(result):
        if not row.any():
            max_idx = arr[i].argmax()
            row[max_idx] = 1
    return result

def convert_array_to_labels(array):
    labels = []
    for idx, value in enumerate(array):
        if value == 1:
            labels.append(idx)
    return labels

def convert_list_of_arrays_to_labels(list_of_arrays):
    all_labels = []
    for array in list_of_arrays:
        labels = convert_array_to_labels(array)
        all_labels.append(labels)
    return all_labels

def compute_metrics_sigmoid(p):

    def threshold_and_max(arr, threshold=0.5):
        result = (arr > threshold)
        for i, row in enumerate(result):
            if not row.any():
                max_idx = arr[i].argmax()
                row[max_idx] = 1
        return result

    preds = threshold_and_max(torch.sigmoid(torch.tensor(p.predictions)), threshold=0.5)

    labels = torch.tensor(p.label_ids)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='weighted')
    # Compute loss using raw logits and labels
    logits = torch.tensor(p.predictions)
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(logits, labels.float()).item()
    return {
        'accuracy': acc,
        'f1': f1,
        'loss': loss
    }

def convert_to_one_hot(labels, num_classes=4):
    """Convert integer labels to one-hot encoding for multi-label classification."""
    # If labels is a list of lists, convert it to a tensor
    if isinstance(labels, list):
        labels = torch.tensor(labels)
    
    # Create a zero tensor of shape (num_samples, num_classes)
    one_hot_labels = torch.zeros((labels.size(0), num_classes))
    
    # Set the appropriate elements to 1
    for i, label_set in enumerate(labels):
        if label_set==7:
            print(f"Label set for index {i}: {label_set}")
        one_hot_labels[i] = F.one_hot(label_set, num_classes=num_classes)
    
    return one_hot_labels.float()


def convert_labels_to_one_hot(data, num_classes=5):
    converted_data = []
    for item in data:
        image, label = item
        label_name = label_names()[label]
        if '&' in label_name:
            label_index = []
            for label in label_name.split('&'):
                if label == 'corrosion':
                    label_index.append(4)
                else:
                    label_index.append(list(label_names().values()).index(label))
            label_one = convert_to_one_hot(label_index, num_classes=num_classes)
            label_one = label_one.sum(axis=0)
            # make it a tensor
            # label_one = torch.tensor(label_one, dtype=torch.float32)
        else:
            if label_name == 'corrosion':
                label_index = [4]
            else:
                label_index = [label]
            label_one = convert_to_one_hot(label_index, num_classes=num_classes)
            label_one = label_one.sum(axis=0)
            # make it a tensor
            # label_one = torch.tensor(label_one, dtype=torch.float32)
        converted_data.append((image, label_one))
    return converted_data


def calculate_class_accuracy_one_hot(true_labels, pred_labels, class_label, threshold=0.5):
    # pred_labels = (pred_logits > threshold) #.astype(int)


    # Get the indices of the samples belonging to the specific class
    class_indices = np.where(np.array(true_labels)[:, class_label] == 1)[0]
    
    # Get the true and predicted labels for the specific class
    class_true_labels = np.array(true_labels)[class_indices, class_label]
    class_pred_labels = np.array(pred_labels)[class_indices, class_label]
    
    # Calculate the accuracy for the specific class
    class_accuracy = accuracy_score(class_true_labels, class_pred_labels)
    
    label_name = label_names(flag='Website')[class_label]
    return label_name, class_accuracy, len(class_indices)

# def calculate_class_accuracy(true_labels, pred_labels, class_label):
#     # Get the indices of the samples belonging to the specific class
#     class_indices = np.where(np.array(true_labels) == class_label)[0]
    
#     # Get the true and predicted labels for the specific class
#     class_true_labels = np.array(true_labels)[class_indices]
#     class_pred_labels = pred_labels[class_indices]
    
#     # Calculate the accuracy for the specific class
#     class_accuracy = accuracy_score(class_true_labels, class_pred_labels)
#     return class_accuracy

def normalize_image(image):
    # Normalize the image to the range [0, 1]
    image_np = np.array(image)
    norm = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    return (norm).astype(np.uint8)


def augment_underrepresented_classes(datass):
    if type(datass) == list:
        datas = datass.copy() 
    else:
        # Convert data format if needed
        converted_datas = []
        for data in datass:
            if isinstance(data, dict):
                if 'images' in data and 'labels' in data:
                    # Convert from {'images': ..., 'labels': ..., 'channels': ...} 
                    # to {0: ..., 1: ..., 2: ...} format
                    converted_item = {
                        0: data['images'],
                        1: data['labels'],
                        2: data.get('channels', None)
                    }
                    converted_datas.append(converted_item)
                else:
                    # Already in correct format or handle tuple format
                    converted_datas.append(data)
            else:
                # Handle tuple format (image, label)
                if len(data) == 2:
                    converted_item = {0: data[0], 1: data[1]}
                    converted_datas.append(converted_item)
                else:
                    converted_datas.append(data)
        
        datas = converted_datas

    print("Augmenting underrepresented classes...")
    print("Augmenting underrepresented classes...")
    print("Augmenting underrepresented classes...")
    label_counts = {label: 0 for label in range(len(datas[0][1]))}
    for data in datas:
        for label, value in enumerate(data[1]):
            if value == 1:
                label_counts[label] += 1

    print(label_counts)
    augmented_data = []
    # Exclude class 0 from the average count calculation
    avg_count = np.max([count for label, count in label_counts.items()])
    threshold = avg_count * 0.6  # Set threshold to 60% of the average count

    for label, count in label_counts.items():
        if count==0:
            continue
        if count < threshold:
            samples_to_augment = [data for data in datas if data[1][label] == 1]
            num_augmentations = min(int(threshold - count), int(count*4))
            #choose smaller
            print('num_augmentations', num_augmentations, label, count, threshold)
            for _ in range(num_augmentations):
                sample = random.choice(samples_to_augment)
                augmented_image = improved_augmentation_fn(sample[0])


                # Randomly choose between the two augmentation functions
                # if np.random.rand() > 0.5:
                # else:
                #     augmented_image = augmentation_fn_combine(image, datas, label)
                augmented_data.append((augmented_image, sample[1]))

                # # Visualize original and augmented images
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # img = sample[0]
                # aug_img = augmented_image.copy()
                # # Convert img for display (grayscale)
                # if isinstance(img, torch.Tensor):
                #     img = img.clone().detach().cpu().numpy()
                #     # If image has shape (C, H, W), take first channel
                #     if img.ndim == 3:
                #         img = img[0]  # Take first channel for grayscale
                #         aug_img = aug_img[0]  # Take first channel for grayscale
                #     elif img.ndim == 2:
                #         pass  # Already H, W
                #     else:
                #         img = np.squeeze(img)
                #     # Normalize to [0, 1]
                #     img_min = img.min()
                #     img_max = img.max()
                #     if img_max > img_min:
                #         img = (img - img_min) / (img_max - img_min)
                #         aug_img = (aug_img - img_min) / (img_max - img_min)
                #     else:
                #         img = np.zeros_like(img)

                # axes[0].imshow(img, cmap='gray')
                # axes[0].set_title('Original Image')
                # axes[0].axis('off')
                # axes[1].imshow(aug_img, cmap='gray')
                # axes[1].set_title('Augmented Image')
                # axes[1].axis('off')
                # os.makedirs('./Augmented1', exist_ok=True)
                # plt.savefig(f'./Augmented1/augmented_image_{label}_{_}.png')
                # plt.close()

    datas.extend(augmented_data)
    return datas

def augmentation_fn(image):
    if np.random.rand() > 0.5:
        image = torch.flip(image, dims=[2])  # Flip along the width dimension

    fold = np.random.choice([0, 1, 2, 3])
    image = torch.rot90(image, k=fold, dims=(1, 2))
    # # # Random contrast enhancement
    # image = image.convert('L')
    # enhancer = ImageEnhance.Contrast(image)
    # image = enhancer.enhance(np.random.uniform(0.8, 1.2))

    return image
def improved_augmentation_fn(image):
    # Apply geometric transforms
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-1]) # Horizontal flip
    
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-2]) # Vertical flip
    
    fold = random.choice([0, 1, 2, 3])
    image = torch.rot90(image, k=fold, dims=(-2, -1))
    
    # Apply photometric transforms
    brightness_factor = random.uniform(0.99, 1.01)
    contrast_factor = random.uniform(0.99, 1.01)
    # Use RandomAffine for a combination of rotations, translations, and shearing
    # This provides more diverse and realistic variations
    affine_transform = RandomAffine(
        degrees=(-0.5, 0.5),        # Random rotation between -30 and 30 degrees
        translate=(0.03, 0.03),     # Random horizontal and vertical translation up to 3%
        shear=(-1, 1)           # Random shearing between -15 and 15 degrees
    )
    image = affine_transform(image)

    # Assuming the image is a PyTorch tensor
    # Note: These operations might need to be adjusted based on image tensor shape
    image = image * brightness_factor

    
    # Add Gaussian noise
    # Add Gaussian noise with mean 0 and a chosen standard deviation
    noise_std = torch.rand(1) * 0.03
    noise = torch.randn_like(image) * noise_std
    image = image + noise
    
    # Clip values to stay within a valid range
    # image = torch.clamp(image, 0, 255)
    
    return image

def combine_images(image1, image2):
    # Resize images to the same size
    image1 = image1.resize((256, 256))
    image2 = image2.resize((256, 256))
    
    # Convert images to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    # Combine images by averaging their pixel values
    combined_arr = (arr1.astype(np.float32) + arr2.astype(np.float32)) / 2
    combined_arr = combined_arr.astype(np.uint8)
    
    return Image.fromarray(combined_arr)

def augmentation_fn_combine(image, data, label):
    
    # Combine with another random image from the same class
    same_class_images = [d[0] for d in data if d[1] == label]
    if same_class_images:
        other_image = Image.fromarray(random.choice(same_class_images))
        image = combine_images(image, other_image)
    
    return image



def label_names(flag=''):
    if flag=='Website':
                return {
            0: 'good',
            1: 'crack',
            2: 'cross',
            3: 'dark',
            4: 'corrosion',
            5: 'discoloration',
            6: 'delamination',
        }
    else:
        return {
            0: 'good',
            1: 'crack',
            2: 'cross',
            3: 'dark',
            4: 'crack&cross',
            5: 'crack&dark',
            6: 'crack&cross&dark',
            7: 'corrosion',
            8: 'corrosion&cross',
            9: 'corrosion&crack&cross',
            10: 'corrosion&crack'
        }


def count_data_per_multiclass(data):
    label_counts = {label: 0 for label in label_names().keys()}
    label_namessss = {label: 0 for label in label_names().values()}
    class_names = label_names()

    for _, label in data:
        label_counts[label] += 1
        label_namessss[class_names[label]] += 1 
    # Drop labels with 0 count
    label_counts = {label: count for label, count in label_counts.items() if count > 0}
    label_namessss = {label: count for label, count in label_namessss.items() if count > 0}
    print(label_counts)
    print(label_namessss)
          
def count_data_per_class(data):
    label_counts = {label: 0 for label in label_names().keys()}
    label_namessss = {label: 0 for label in label_names().values()}

    class_names = label_names()

    for _, label in data:
        label_name = class_names[label]
        if '&' in label_name:
            individual_labels = label_name.split('&')
            for individual_label in individual_labels:
                individual_label_index = list(class_names.values()).index(individual_label)
                label_counts[individual_label_index] += 1
                label_namessss[class_names[individual_label_index]] += 1 
        else:
            label_counts[label] += 1
            label_namessss[class_names[label]] += 1 
    # Drop labels with 0 count
    label_counts = {label: count for label, count in label_counts.items() if count > 0}
    label_namessss = {label: count for label, count in label_namessss.items() if count > 0}
    print(label_counts)
    print(label_namessss)
    return label_counts


def count_data_per_class_in_labels(labelss):
    label_counts = {label: 0 for label in label_names(flag='Website').keys()}
    label_namessss = {label: 0 for label in label_names(flag='Website').values()}

    class_names = label_names(flag='Website')

    for labels in labelss:
        for label in labels:
            label_counts[label] += 1
            label_namessss[class_names[label]] += 1 
    # Drop labels with 0 count
    label_counts = {label: count for label, count in label_counts.items() if count > 0}
    label_namessss = {label: count for label, count in label_namessss.items() if count > 0}
    print(label_counts)
    print(label_namessss)
    return label_counts
        
    # Print the counts and their label names
    for label, count in label_counts.items():
        if count>0:
            print(f"{class_names[label]}: {count}")
    return label_counts

# def is_prime(n):
#     """Check if a number is prime."""
#     if n <= 1:
#         return False
#     for i in range(2, int(math.sqrt(n)) + 1):
#         if n % i == 0:
#             return False
#     return True

def find_optimal_grid(n):
    """Return grid dimensions (rows, cols) for plotting n images, aiming for a shape as square as possible."""
    best_diff = n  # Initialize with a large difference
    best_rows, best_cols = 1, n
    for rows in range(1, int(math.sqrt(n)) + 2):
        cols = math.ceil(n / rows)
        if rows * cols >= n:
            diff = abs(rows - cols)
            if diff < best_diff:
                best_diff = diff
                best_rows, best_cols = rows, cols
    return best_rows, best_cols


def class_label_save(predlabels, im_names, label_names,  file_name='predictions_tiso.parquet'):
    label_counts = {}
    for label in predlabels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    label_counts = dict(sorted(label_counts.items()))
    print(label_counts)
    output_dir='./Data/parquet'
    label_counts_named = {(label_names)[int_label]: count for int_label, count in label_counts.items()}
    print(label_counts_named)
    output_predictions_path = os.path.join(output_dir, file_name)
    predictions_df = pd.DataFrame(predlabels, index=im_names)
    a = predictions_df.index.str.extract(r'(Cell\d+)')
    a = pd.DataFrame(a.values, index=predictions_df.index)
    b = pd.concat([a, predictions_df],axis=1)
    b.index = b.index.str.replace(r'_Cell\d+', '', regex=True)
    b.index = b.index.str.replace(r'_EL', '', regex=True)
    b.index = b.index.map(lambda x: x.replace('.tif', '') if isinstance(x, str) else x)
    try:
        b.index = b.index.astype(int)
    except ValueError:
        pass
    b.columns = ['cells','classes']
    b = b.pivot(columns='cells')
    b.to_parquet(output_predictions_path, index=True)
    print(f"Predictions saved to {output_predictions_path}")


def logits_to_classes(logits, initial_threshold=0.2, relative_threshold=0.7, min_confidence=0.1):
    """
    Advanced adaptive thresholding with multiple fallback strategies.
    
    Args:
        logits: The logits output by the model.
        initial_threshold: Primary threshold for classification.
        relative_threshold: Percentage of max probability to use as backup threshold.
        min_confidence: Minimum confidence required for any prediction.
    
    Returns:
        np.ndarray: Predicted class indices.
    """
    probabilities = torch.sigmoid(torch.tensor(logits[0])).numpy()
    predicted_classes = np.zeros_like(probabilities, dtype=int)
    
    for i, prob in enumerate(probabilities):
        pred = (prob > initial_threshold).astype(int)
        
        if pred.any():
            # Strategy 1: Use initial threshold
            predicted_classes[i] = pred
        else:
            # Strategy 2: Use relative threshold (X% of max probability)
            max_prob = prob.max()
            if max_prob >= min_confidence:
                relative_thresh = max_prob * relative_threshold
                pred_relative = (prob >= relative_thresh).astype(int)
                
                if pred_relative.any():
                    predicted_classes[i] = pred_relative
                else:
                    # Strategy 3: Predict only the maximum if it meets minimum confidence
                    max_idx = prob.argmax()
                    pred[max_idx] = 1
                    predicted_classes[i] = pred
            # If max_prob < min_confidence, leave as all zeros
    
    return predicted_classes, probabilities

def logits_to_classes_TISO(logits, initial_threshold=0.9, relative_threshold=0.7, min_confidence=0.5):
    """
    Advanced adaptive thresholding with multiple fallback strategies.
    
    Args:
        logits: The logits output by the model.
        initial_threshold: Primary threshold for classification.
        relative_threshold: Percentage of max probability to use as backup threshold.
        min_confidence: Minimum confidence required for any prediction.
    
    Returns:
        np.ndarray: Predicted class indices.
    """
    probabilities = torch.sigmoid(torch.tensor(logits[0])).numpy()
    predicted_classes = np.zeros_like(probabilities, dtype=int)
    
    for i, prob in enumerate(probabilities):
        pred = (prob > initial_threshold).astype(int)
        
        if pred.any():
            # Strategy 1: Use initial threshold
            predicted_classes[i] = pred
        else:
            # Strategy 2: Use relative threshold (X% of max probability)
            max_prob = prob.max()
            if max_prob >= min_confidence:
                relative_thresh = max_prob * relative_threshold
                pred_relative = (prob >= relative_thresh).astype(int)
                
                if pred_relative.any():
                    predicted_classes[i] = pred_relative
                else:
                    # Strategy 3: Predict only the maximum if it meets minimum confidence
                    max_idx = prob.argmax()
                    pred[max_idx] = 1
                    predicted_classes[i] = pred
            # If max_prob < min_confidence, leave as all zeros
    # Count how many predicted_classes rows are all zeros (empty predictions)
    empty_count = np.sum([np.all(row == 0) for row in predicted_classes])
    print(f"Number of empty predictions: {empty_count}")
    return predicted_classes


def logits_to_classes_old(logits, initial_threshold=0.2):
    """
    Convert logits to class predictions for multi-label classification with an adaptive threshold.

    Args:
        logits (torch.Tensor or np.ndarray): The logits output by the model.
        initial_threshold (float): The initial threshold to apply to the probabilities to determine class membership.

    Returns:
        np.ndarray: An array of predicted class indices.
    """
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(torch.tensor(logits[0])).numpy()
    print(probabilities)
    # Initialize the predicted classes array
    predicted_classes = np.zeros_like(probabilities, dtype=int)

    for i, prob in enumerate(probabilities):
        # Apply the initial threshold
        pred = (prob > initial_threshold).astype(int)
        # Print values in pred with .2f format
        print("Pred values: [" + ", ".join(f"{v:.2f}" for v in prob) + "]")
        predicted_classes[i] = pred
    
    return predicted_classes #predicted_classes !!!!! CHANGED

def find_mismatched_data(temp_Infinity, trainer, data_Infinity, current_dir, images_folder):
    
    # #################################################  PREDICTIONS validation #########################################################
    # tensor_label_list_Infinity = just_transform(temp_Infinity, channels=[0], name='infinity')
    dataset_infint   = PVDataset(temp_Infinity,   channels=[[0]]*len(temp_Infinity),   scale=1, return_labels=True)

    predictions_inf = trainer.predict(dataset_infint) 
    # Save predictions to a file
    np.save('predictions_inf10.npy', np.array(predictions_inf, dtype=object))
    pred_labels, _ = logits_to_classes(predictions_inf, initial_threshold=0.5)
    # Save pred_labels to a file
    true_labels_inf = np.array([item['labels'] for item in dataset_infint])
    class_accuracies = {}
    for label in range(7):
        label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels_inf, pred_labels, class_label=label)
        print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")

    # predictions = np.load('predictions_inf10.npy', allow_pickle=True)
    # pred_labels, _ = logits_to_classes(predictions, initial_threshold=0.5)
    
    
    # If your labels are one-hot or multi-hot arrays:
    matching_indices = [i for i, (t, p) in enumerate(zip(true_labels_inf, pred_labels)) if np.array_equal(t, p)]
    non_matching_indices = []
    for i, (t, p) in enumerate(zip(true_labels_inf, pred_labels)):
        # Check if any element in t matches any element in p (at any position)
        if not np.any(np.isin(np.where(t == 1)[0], np.where(p == 1)[0])):
            non_matching_indices.append(i)

    # Select the matching images and labels
    matching_images_labels = [data_Infinity[i] for i in matching_indices]
    non_matching_images_labels = [data_Infinity[i] for i in non_matching_indices]

    labels_as_integers = [np.where(label == 1)[0].tolist() for _, label in matching_images_labels]
    save_images_by_label(matching_images_labels, labels_as_integers, current_dir+images_folder+'/Infinity_images_cleaner/')
    # Save non-matching images
    labels_as_integers_non = [np.where(label == 1)[0].tolist() for _, label in non_matching_images_labels]
    save_images_by_label(non_matching_images_labels, labels_as_integers_non, current_dir+images_folder+'/Infinity_images_wrong/')

    labels_as_integers = [np.where(label == 1)[0].tolist() for label in true_labels_inf]
    save_images_by_label(data_Infinity, labels_as_integers, current_dir+images_folder+'/Infinity_images_original/')

    # plot_samples_from_all_labels_with_acc(dataset_infint, pred_labels, class_accuracies, data_name='Infinity', outfolder='./Data')
    exit()


def verify_data_normalization(dataset, sample_size=100, tolerance=0.1, verbose=True):
    """
    Verify that data in a dataset is properly normalized to mean ~0 and std ~1.
    
    Args:
        dataset: PyTorch dataset or DataLoader
        sample_size: Number of samples to check (None for all samples)
        tolerance: Acceptable deviation from target mean (0) and std (1)
        verbose: Whether to print detailed statistics
    
    Returns:
        dict: Statistics about the normalization
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    
    # Handle different input types
    if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
        # It's a dataset
        if sample_size and sample_size < len(dataset):
            # Sample random subset
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            subset = Subset(dataset, indices)
            dataloader = DataLoader(subset, batch_size=32, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    else:
        # Assume it's already a DataLoader
        dataloader = dataset
    
    all_images = []
    all_channels_stats = []
    
    print(f"Checking normalization for {len(dataloader)} batches...")
    
    # Collect all images
    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict) and 'images' in batch:
            images = batch['images']
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
            
        # Handle different image formats
        if len(images.shape) == 5:  # (batch, channels, depth, height, width) - multi-channel
            images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        elif len(images.shape) == 4:  # (batch, channels, height, width)
            pass
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
        
        all_images.append(images)
        
        if verbose and batch_idx < 3:  # Show first 3 batches
            batch_mean = images.mean().item()
            batch_std = images.std().item()
            print(f"Batch {batch_idx}: shape={images.shape}, mean={batch_mean:.4f}, std={batch_std:.4f}")
    
    # Concatenate all images
    all_images = torch.cat(all_images, dim=0)
    print(f"Total images collected: {all_images.shape}")
    
    # Calculate overall statistics
    overall_mean = all_images.mean().item()
    overall_std = all_images.std().item()
    
    # Calculate per-channel statistics
    num_channels = all_images.shape[1]
    channel_stats = []
    
    for c in range(num_channels):
        channel_data = all_images[:, c, :, :]
        channel_mean = channel_data.mean().item()
        channel_std = channel_data.std().item()
        channel_stats.append({
            'channel': c,
            'mean': channel_mean,
            'std': channel_std,
            'mean_ok': abs(channel_mean) < tolerance,
            'std_ok': abs(channel_std - 1.0) < tolerance
        })
    
    # Check if normalization is acceptable
    mean_ok = abs(overall_mean) < tolerance
    std_ok = abs(overall_std - 1.0) < tolerance
    all_channels_ok = all([c['mean_ok'] and c['std_ok'] for c in channel_stats])
    
    results = {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'target_mean': 0.0,
        'target_std': 1.0,
        'tolerance': tolerance,
        'mean_ok': mean_ok,
        'std_ok': std_ok,
        'all_channels_ok': all_channels_ok,
        'channel_stats': channel_stats,
        'total_samples': all_images.shape[0],
        'image_shape': all_images.shape[1:],
        'recommendation': None
    }
    
    # Generate recommendations
    if not mean_ok or not std_ok:
        if not mean_ok and not std_ok:
            results['recommendation'] = f"Data needs normalization: mean={overall_mean:.4f} (should be ~0), std={overall_std:.4f} (should be ~1)"
        elif not mean_ok:
            results['recommendation'] = f"Data needs mean centering: mean={overall_mean:.4f} (should be ~0)"
        else:
            results['recommendation'] = f"Data needs std scaling: std={overall_std:.4f} (should be ~1)"
    else:
        results['recommendation'] = "Data normalization looks good!"
    
    if verbose:
        print("\n" + "="*60)
        print("DATA NORMALIZATION CHECK RESULTS")
        print("="*60)
        print(f"Total samples checked: {results['total_samples']}")
        print(f"Image shape: {results['image_shape']}")
        print(f"Tolerance: ±{tolerance}")
        print(f"\nOverall Statistics:")
        print(f"  Mean: {overall_mean:.6f} (target: 0.0) {'✓' if mean_ok else '✗'}")
        print(f"  Std:  {overall_std:.6f} (target: 1.0) {'✓' if std_ok else '✗'}")
        
        print(f"\nPer-Channel Statistics:")
        for stat in channel_stats:
            mean_status = '✓' if stat['mean_ok'] else '✗'
            std_status = '✓' if stat['std_ok'] else '✗'
            print(f"  Channel {stat['channel']}: mean={stat['mean']:.6f} {mean_status}, std={stat['std']:.6f} {std_status}")
        
        print(f"\nRecommendation: {results['recommendation']}")
        print("="*60)
    
    return results

# Additional helper function to check transforms
def check_transform_normalization(transform, sample_image):
    """
    Check if a transform properly normalizes a sample image.
    
    Args:
        transform: Torchvision transform
        sample_image: PIL Image or tensor
    
    Returns:
        dict: Before/after statistics
    """
    import torch
    from torchvision.transforms import ToPILImage, ToTensor
    
    # Convert to tensor if needed
    if not isinstance(sample_image, torch.Tensor):
        sample_image = ToTensor()(sample_image)
    
    # Apply transform
    if sample_image.dim() == 3:
        sample_image = sample_image.unsqueeze(0)  # Add batch dimension
    
    transformed = transform(sample_image.squeeze(0))
    if transformed.dim() == 3:
        transformed = transformed.unsqueeze(0)
    
    before_stats = {
        'mean': sample_image.mean().item(),
        'std': sample_image.std().item(),
        'min': sample_image.min().item(),
        'max': sample_image.max().item()
    }
    
    after_stats = {
        'mean': transformed.mean().item(),
        'std': transformed.std().item(),
        'min': transformed.min().item(),
        'max': transformed.max().item()
    }
    
    print("Transform Normalization Check:")
    print(f"Before: mean={before_stats['mean']:.4f}, std={before_stats['std']:.4f}, range=[{before_stats['min']:.4f}, {before_stats['max']:.4f}]")
    print(f"After:  mean={after_stats['mean']:.4f}, std={after_stats['std']:.4f}, range=[{after_stats['min']:.4f}, {after_stats['max']:.4f}]")
    
    return {'before': before_stats, 'after': after_stats}


def check_raw_tensor_normalization(tensor_data_list, data_name, check_all=True):
    """
    Check normalization of raw tensor data before transforms

    Args:
        tensor_data_list: List of (tensor, label) tuples, (tensor,) tuples, or just tensors
        data_name: Name for printing
        check_all: If True, check all data; if False, sample
    """
    print(f"\n--- Checking Raw {data_name} Data Normalization ---")

    if not tensor_data_list:
        print(f"No data in {data_name}")
        return

    all_tensors = []

    # Process all data or sample
    data_to_check = tensor_data_list if check_all else tensor_data_list[:min(100, len(tensor_data_list))]

    for i, entry in enumerate(data_to_check):
        if i % 1000 == 0 and i > 0:  # Progress indicator for large datasets
            print(f"  Processed {i}/{len(data_to_check)} samples...")

        # Accept (tensor, label), (tensor,), or tensor
        if isinstance(entry, tuple):
            tensor = entry[0]
        else:
            tensor = entry

        # Handle different tensor shapes
        if len(tensor.shape) == 2:  # (H, W)
            tensor = tensor.unsqueeze(0)  # Add channel dimension -> (1, H, W)
        elif len(tensor.shape) == 3:  # (C, H, W)
            pass
        else:
            print(f"  Warning: Unexpected tensor shape {tensor.shape} at index {i}")
            continue

        all_tensors.append(tensor.float())

    if not all_tensors:
        print(f"  No valid tensors found in {data_name}")
        return

    # Stack all tensors
    try:
        all_data = torch.stack(all_tensors)
        print(f"  Stacked tensor shape: {all_data.shape}")
    except Exception as e:
        print(f"  Error stacking tensors: {e}")
        # Try with different approach - flatten and check
        flattened_data = torch.cat([t.flatten() for t in all_tensors])
        all_data = flattened_data.view(-1, 1, 1, 1)  # Reshape for consistent processing

    # Calculate statistics
    overall_mean = all_data.mean().item()
    overall_std = all_data.std().item()
    overall_min = all_data.min().item()
    overall_max = all_data.max().item()

    # Per-channel statistics
    num_channels = all_data.shape[1]
    print(f"  Total samples: {all_data.shape[0]}")
    print(f"  Number of channels: {num_channels}")
    print(f"  Data type: {all_data.dtype}")

    print(f"\n  Overall Statistics:")
    print(f"    Mean: {overall_mean:.6f}")
    print(f"    Std:  {overall_std:.6f}")
    print(f"    Min:  {overall_min:.6f}")
    print(f"    Max:  {overall_max:.6f}")
    print(f"    Range: [{overall_min:.6f}, {overall_max:.6f}]")

    # Check if data looks normalized
    mean_ok = abs(overall_mean) < 0.1
    std_ok = abs(overall_std - 1.0) < 0.2  # More lenient for raw data

    if num_channels > 1:
        print(f"\n  Per-Channel Statistics:")
        for c in range(num_channels):
            channel_data = all_data[:, c, :, :]
            ch_mean = channel_data.mean().item()
            ch_std = channel_data.std().item()
            ch_min = channel_data.min().item()
            ch_max = channel_data.max().item()
            print(f"    Channel {c}: mean={ch_mean:.6f}, std={ch_std:.6f}, range=[{ch_min:.4f}, {ch_max:.4f}]")

    # Provide assessment
    if overall_min >= 0 and overall_max <= 1:
        print(f"  ✓ Data appears to be in [0, 1] range")
    elif overall_min >= 0 and overall_max <= 255:
        print(f"  → Data appears to be in [0, 255] range (needs normalization)")
    elif mean_ok and std_ok:
        print(f"  ✓ Data appears to be normalized (mean≈0, std≈1)")
    else:
        print(f"  ⚠ Data may need normalization")

    return {
        'mean': overall_mean,
        'std': overall_std,
        'min': overall_min,
        'max': overall_max,
        'shape': all_data.shape,
        'channels': num_channels,
        'assessment': 'normalized' if mean_ok and std_ok else 'needs_normalization'
    }

def check_component_normalization(data, sample_size=50):
    """Check actual normalization values for EL, VIS, and UV components"""
    
    el_pixels = []
    vis_pixels = [[], [], []]  
    uv_pixels = [[], [], []]   
    
    # Define the transforms used in just_transform
    el_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    rgb_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Checking component normalization...")
    sample_data = data[:min(sample_size, len(data))]
    
    for img, _ in tqdm(sample_data, desc="Processing samples"):
        if img.shape[-1] == 7:
            # Extract components (same as in just_transform)
            el = img[:, :, 0]        # (H, W)
            vis = img[:, :, 1:4]     # (H, W, 3)
            uv = img[:, :, 4:7]      # (H, W, 3)
            
            # Apply the same transforms
            el_transformed = el_transform(el)      # (1, 224, 224)
            vis_transformed = rgb_transform(vis)   # (3, 224, 224)
            uv_transformed = rgb_transform(uv)     # (3, 224, 224)
            
            # Collect pixels
            el_pixels.extend(el_transformed.flatten().tolist())
            
            for ch in range(3):
                vis_pixels[ch].extend(vis_transformed[ch].flatten().tolist())
                uv_pixels[ch].extend(uv_transformed[ch].flatten().tolist())
    
    # Calculate statistics
    def calc_stats(pixels, name):
        if pixels:
            pixels_array = np.array(pixels)
            mean_val = pixels_array.mean()
            std_val = pixels_array.std()
            min_val = pixels_array.min()
            max_val = pixels_array.max()
            
            print(f"\n{name}:")
            print(f"  Mean: {mean_val:.6f}")
            print(f"  Std:  {std_val:.6f}")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            
            # Check if normalized
            mean_ok = abs(mean_val) < 0.1
            std_ok = abs(std_val - 1.0) < 0.2
            status = "✓ Good" if mean_ok and std_ok else "✗ Needs attention"
            print(f"  Status: {status}")
            
            return mean_val, std_val, min_val, max_val
        return None, None, None, None
    
    print("\n" + "="*60)
    print("COMPONENT NORMALIZATION ANALYSIS")
    print("="*60)
    
    # EL analysis
    calc_stats(el_pixels, "EL Component (Channel 0)")
    
    # VIS analysis
    for ch in range(3):
        calc_stats(vis_pixels[ch], f"VIS Component Channel {ch+1}")
    
    # UV analysis  
    for ch in range(3):
        calc_stats(uv_pixels[ch], f"UV Component Channel {ch+4}")
    
    print("="*60)
    
    # Overall component analysis
    all_vis = [pixel for ch_pixels in vis_pixels for pixel in ch_pixels]
    all_uv = [pixel for ch_pixels in uv_pixels for pixel in ch_pixels]
    
    print("\nOVERALL COMPONENT SUMMARY:")
    calc_stats(el_pixels, "All EL pixels")
    calc_stats(all_vis, "All VIS pixels")
    calc_stats(all_uv, "All UV pixels")
    
    return {
        'el': el_pixels,
        'vis': vis_pixels, 
        'uv': uv_pixels
    }


def calculate_raw_data_means(data, data_name, sample_size=None):
    """Calculate mean pixel values from raw image data before any transforms"""
    print(f"\nCalculating raw pixel means for {data_name}...")
    
    if sample_size is None:
        sample_data = data
    else:
        sample_data = data[:min(sample_size, len(data))]
    
    all_pixels = []
    
    for i, (img, label) in enumerate(sample_data):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(sample_data)} images...")
        
        # Convert image to float and collect pixels
        if hasattr(img, 'dtype') and img.dtype == np.uint8:
            img_float = img.astype(np.float32)
        else:
            img_float = img
            
        all_pixels.extend(img_float.flatten())
    
    # Calculate statistics
    pixels_array = np.array(all_pixels)
    mean_val = pixels_array.mean()
    std_val = pixels_array.std()
    min_val = pixels_array.min()
    max_val = pixels_array.max()
    
    print(f"\n{data_name} Raw Data Statistics:")
    print(f"  Total pixels: {len(pixels_array):,}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  Data type: {type(pixels_array[0])}")
    
    # Determine if data is in [0,255] or [0,1] range
    if max_val > 1:
        print(f"  → Data appears to be in [0, 255] range (uint8)")
        normalized_mean = mean_val / 255.0
        normalized_std = std_val / 255.0
        print(f"  → Normalized to [0,1]: mean={normalized_mean:.4f}, std={normalized_std:.4f}")
    else:
        print(f"  → Data already in [0, 1] range")
        normalized_mean = mean_val
        normalized_std = std_val
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'normalized_mean': normalized_mean,
        'normalized_std': normalized_std,
        'total_pixels': len(pixels_array)
    }


def calculate_raw_data_means(data, data_name, sample_size=None):
    """
    Calculate mean, std, min, max for raw image data.
    For 7-channel Website data, splits into EL, VIS, UV and computes stats per component.
    """
    print(f"\nCalculating raw pixel means for {data_name}...")

    if sample_size is None:
        sample_data = data
    else:
        sample_data = data[:min(sample_size, len(data))]

    # For Website data: split into EL, VIS, UV
    is_website = False
    for img, _ in sample_data:
        if hasattr(img, "shape") and img.shape[-1] == 7:
            is_website = True
            break

    if is_website:
        el_pixels = []
        vis_pixels = [[], [], []]
        uv_pixels = [[], [], []]
        for img, _ in sample_data:
            if img.shape[-1] == 7:
                el = img[:, :, 0]
                vis = img[:, :, 1:4]
                uv = img[:, :, 4:7]
                el_pixels.extend(el.flatten())
                for ch in range(3):
                    vis_pixels[ch].extend(vis[:, :, ch].flatten())
                    uv_pixels[ch].extend(uv[:, :, ch].flatten())
        # Print stats for EL
        el_arr = np.array(el_pixels, dtype=np.float32)
        print(f"\nEL channel: mean={el_arr.mean():.2f}, std={el_arr.std():.2f}, min={el_arr.min():.0f}, max={el_arr.max():.0f}")
        if el_arr.max() > 1:
            print(f"  Normalized: mean={el_arr.mean()/255:.4f}, std={el_arr.std()/255:.4f}")
        # Print stats for VIS
        for ch in range(3):
            vis_arr = np.array(vis_pixels[ch], dtype=np.float32)
            print(f"VIS channel {ch}: mean={vis_arr.mean():.2f}, std={vis_arr.std():.2f}, min={vis_arr.min():.0f}, max={vis_arr.max():.0f}")
            if vis_arr.max() > 1:
                print(f"  Normalized: mean={vis_arr.mean()/255:.4f}, std={vis_arr.std()/255:.4f}")
        # Print stats for UV
        for ch in range(3):
            uv_arr = np.array(uv_pixels[ch], dtype=np.float32)
            print(f"UV channel {ch}: mean={uv_arr.mean():.2f}, std={uv_arr.std():.2f}, min={uv_arr.min():.0f}, max={uv_arr.max():.0f}")
            if uv_arr.max() > 1:
                print(f"  Normalized: mean={uv_arr.mean()/255:.4f}, std={uv_arr.std()/255:.4f}")
        return {
            'el': el_arr,
            'vis': [np.array(vis_pixels[ch], dtype=np.float32) for ch in range(3)],
            'uv': [np.array(uv_pixels[ch], dtype=np.float32) for ch in range(3)]
        }
    else:
        # For single-channel data
        all_pixels = []
        for img, label in sample_data:
            if hasattr(img, 'dtype') and img.dtype == np.uint8:
                img_float = img.astype(np.float32)
            else:
                img_float = img
            all_pixels.extend(img_float.flatten())
        pixels_array = np.array(all_pixels)
        mean_val = pixels_array.mean()
        std_val = pixels_array.std()
        min_val = pixels_array.min()
        max_val = pixels_array.max()
        print(f"\n{data_name} Raw Data Statistics:")
        print(f"  Total pixels: {len(pixels_array):,}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std: {std_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Data type: {type(pixels_array[0])}")
        if max_val > 1:
            print(f"  → Data appears to be in [0, 255] range (uint8)")
            normalized_mean = mean_val / 255.0
            normalized_std = std_val / 255.0
            print(f"  → Normalized to [0,1]: mean={normalized_mean:.4f}, std={normalized_std:.4f}")
        else:
            print(f"  → Data already in [0, 1] range")
            normalized_mean = mean_val
            normalized_std = std_val
        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'normalized_mean': normalized_mean,
            'normalized_std': normalized_std,
            'total_pixels': len(pixels_array)
        }
    
def calculate_per_channel_stats(data, resize_to=(224, 224)):
    """
    Calculate the mean and standard deviation for each channel across the entire dataset.
    Accepts either a list of (img, label) tuples, (img,) tuples, or a list of images.
    All images are resized to `resize_to` before stacking.
    Handles multi-channel images (e.g., 7-channel) by resizing each channel separately.
    """
    print("Calculating per-channel statistics across the entire dataset...")

    image_tensors = []
    resize_transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor()
    ])

    for entry in tqdm(data, desc="Stacking images"):
        # Accept (img, label), (img,), or img only
        if isinstance(entry, tuple):
            if len(entry) == 2:
                img = entry[0]
            elif len(entry) == 1:
                img = entry[0]
            else:
                img = entry
        else:
            img = entry

        if isinstance(img, np.ndarray):
            # If shape is (H, W, C) and C > 4, process each channel separately
            if img.ndim == 3 and img.shape[-1] > 4:
                channels = []
                for ch in range(img.shape[-1]):
                    channel_img = Image.fromarray(img[:, :, ch].astype(np.uint8), mode='L')
                    channel_tensor = resize_transform(channel_img)  # (1, H, W)
                    channels.append(channel_tensor)
                img_tensor = torch.cat(channels, dim=0)  # (C, H, W)
            elif img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_tensor = resize_transform(img_pil)  # (C, H, W)
            elif img.ndim == 2:
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_tensor = resize_transform(img_pil).unsqueeze(0)
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32)
        elif isinstance(img, torch.Tensor):
            img_tensor = img.float()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        image_tensors.append(img_tensor)

    all_images_stacked = torch.stack(image_tensors)

    mean = all_images_stacked.mean(dim=[0, 2, 3])
    std = all_images_stacked.std(dim=[0, 2, 3])

    print(f"Calculated Mean per channel: {mean}")
    print(f"Calculated Std per channel: {std}")

    return mean.tolist(), std.tolist()

def just_transform_with_norm(data, calculated_mean, calculated_std, resize_to=(224, 224)):
    """
    Transform and normalize multi-channel images (e.g., 7-channel Website data).
    Applies resizing and normalization directly to tensors.
    """
    from torchvision.transforms.functional import resize

    tensors = []
    labels = []
    for img, label in tqdm(data, desc="Normalizing images"):
        # Convert to float tensor and permute to (C, H, W)
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[-1] == len(calculated_mean):
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            elif img.ndim == 2:
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32)
        else:
            img_tensor = img.float()

        # Resize each channel separately and stack
        resized_channels = []
        for c in range(img_tensor.shape[0]):
            channel = img_tensor[c, :, :].unsqueeze(0)  # (1, H, W)
            channel_resized = resize(channel, resize_to)  # (1, resize_to[0], resize_to[1])
            resized_channels.append(channel_resized)
        img_tensor_resized = torch.cat(resized_channels, dim=0)  # (C, H, W)

        # Normalize: (x - mean) / std for each channel
        mean = torch.tensor(calculated_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(calculated_std, dtype=torch.float32).view(-1, 1, 1)
        img_tensor_norm = (img_tensor_resized/255. - mean) / std

        tensors.append(img_tensor_norm)
        labels.append(label)

    return list(zip(tensors, labels))




def just_transform_with_norm_without_label(data, calculated_mean, calculated_std, resize_to=(224, 224)):
    """
    Transform and normalize multi-channel images (e.g., 7-channel Website data).
    Applies resizing and normalization directly to tensors.
    """
    from torchvision.transforms.functional import resize

    tensors = []
    for img in tqdm(data, desc="Normalizing images"):
        # Convert to float tensor and permute to (C, H, W)
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[-1] == len(calculated_mean):
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            elif img.ndim == 2:
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32)
        else:
            img_tensor = img.float()



        # Resize each channel separately and stack
        resized_channels = []
        for c in range(img_tensor.shape[0]):
            channel = img_tensor[c, :, :].unsqueeze(0)  # (1, H, W)
            channel_resized = resize(channel, resize_to)  # (1, resize_to[0], resize_to[1])
            resized_channels.append(channel_resized)
        img_tensor_resized = torch.cat(resized_channels, dim=0)  # (C, H, W)

        # Normalize: (x - mean) / std for each channel
        mean = torch.tensor(calculated_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(calculated_std, dtype=torch.float32).view(-1, 1, 1)
        img_tensor_norm = (img_tensor_resized/255. - mean) / std

        tensors.append(img_tensor_norm)

    return list(tensors)

def _ensure_len(lbl, L=7):
    # convert to numpy array
    if hasattr(lbl, "numpy"):
        arr = np.asarray(lbl.numpy())
    else:
        arr = np.asarray(lbl)
    # scalar label -> one-hot
    if arr.ndim == 0:
        out = np.zeros(L, dtype=float)
        out[int(arr)] = 1.0
        return out
    # truncate or pad to length L
    if arr.size >= L:
        return arr[:L].astype(float)
    out = np.zeros(L, dtype=float)
    out[:arr.size] = arr.astype(float)
    return out

