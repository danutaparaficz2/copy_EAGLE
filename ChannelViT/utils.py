from sklearn.metrics import f1_score, accuracy_score, log_loss
import torch
from transformers import  EvalPrediction
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn

import random
import os
import json
from PIL import Image, ImageEnhance, ImageOps
import math



def select_images_by_label(ds, predlabels, label):
    selected_data = []
    selected_predlabels = []
    for idx, s in enumerate(ds):
        if s['labels'][label] == 1:
                selected_data.append(s)
                if predlabels is not None:
                    selected_predlabels.append(predlabels[idx])
                else:
                    selected_predlabels = None
        
    return selected_data, selected_predlabels


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


def augment_underrepresented_classes(datas):
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
    threshold = avg_count * 0.85  # Set threshold to 75% of the average count

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
                augmented_image = augmentation_fn(sample[0])


                # Randomly choose between the two augmentation functions
                # if np.random.rand() > 0.5:
                # else:
                #     augmented_image = augmentation_fn_combine(image, datas, label)
                augmented_data.append({0:augmented_image, 1:sample[1]})

                # Visualize original and augmented images
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # axes[0].imshow(np.array(sample['images'])[0])                
                # axes[0].set_title('Original Image')
                # axes[0].axis('off')
                # axes[1].imshow(np.array(augmented_image))
                # axes[1].set_title('Augmented Image')
                # axes[1].axis('off')
                # # plt.show()
                # # Ensure the folder exists before saving the plot
                # os.makedirs('./Augmented', exist_ok=True)
                # # Save the plot
                # plt.savefig(f'./Augmented/augmented_image_{label}_{_}.png')
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
    """Return the optimal grid dimensions for plotting n images."""
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0 and i>1:
            return i, n // i
        elif n % i == 0 and i==1:
            return (n-1)//2, 2


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

def logits_to_classes(logits, initial_threshold=0.5):
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
    pred_labels = logits_to_classes(predictions_inf, initial_threshold=0.5)
    # Save pred_labels to a file
    true_labels_inf = np.array([item['labels'] for item in dataset_infint])
    class_accuracies = {}
    for label in range(7):
        label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels_inf, pred_labels, class_label=label)
        print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")

    # predictions = np.load('predictions_inf10.npy', allow_pickle=True)
    # pred_labels = logits_to_classes(predictions, initial_threshold=0.5)
    
    
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
