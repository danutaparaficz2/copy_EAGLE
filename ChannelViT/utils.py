from sklearn.metrics import f1_score, accuracy_score, log_loss
import torch
from transformers import  EvalPrediction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

import random
import os
import json
from PIL import Image, ImageEnhance, ImageOps
import math


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
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Initialize the predicted classes array
    predicted_classes = np.zeros_like(probabilities, dtype=int)
    
    for i, prob in enumerate(probabilities):
        # Apply the initial threshold
        pred = (prob > initial_threshold).astype(int)
        
        # If no class is selected, adjust the threshold to select at least one class
        if np.sum(pred) == 0:
            max_prob_index = np.argmax(prob)
            pred[max_prob_index] = 1
        
        predicted_classes[i] = pred
    
    return predicted_classes

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


def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    # loss = log_loss( labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }

def compute_metrics_sigmoid(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)) > 0.5
    labels = torch.tensor(p.label_ids)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1
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
        one_hot_labels[i] = F.one_hot(label_set, num_classes=num_classes)
    
    return one_hot_labels.float()


def convert_labels_to_one_hot(data, num_classes=4):
    converted_data = []
    for item in data:
        image, label = item
        label_name = label_names()[label]
        if '&' in label_name:
            label_index = []
            for label in label_name.split('&'):
                label_index.append(list(label_names().values()).index(label))
            label_one = convert_to_one_hot(label_index, num_classes=num_classes)
            label_one = label_one.sum(axis=0)
        else:
            label_index = [label]
            label_one = convert_to_one_hot(label_index, num_classes=num_classes)
            label_one = label_one.sum(axis=0)
        converted_data.append((image, label_one))
    return converted_data

def plot_samples_from_all_labels(ds, predlabels, unique_labels, data_name='Unknown', outfolder='./Data'):
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

    for label in unique_labels:
        selected_images, selected_predlabels = select_images_by_label(ds, predlabels, label)
        plot_samples_from_specific_label(selected_images, selected_predlabels, label, data_name, outfolder)



def calculate_class_accuracy_one_hot(true_labels, pred_logits, class_label, threshold=0.5):
    pred_labels = (pred_logits > threshold).astype(int)


    # Get the indices of the samples belonging to the specific class
    class_indices = np.where(np.array(true_labels)[:, class_label] == 1)[0]
    
    # Get the true and predicted labels for the specific class
    class_true_labels = np.array(true_labels)[class_indices, class_label]
    class_pred_labels = pred_labels[class_indices, class_label]
    
    # Calculate the accuracy for the specific class
    class_accuracy = accuracy_score(class_true_labels, class_pred_labels)
    return class_accuracy

def calculate_class_accuracy(true_labels, pred_labels, class_label):
    # Get the indices of the samples belonging to the specific class
    class_indices = np.where(np.array(true_labels) == class_label)[0]
    
    # Get the true and predicted labels for the specific class
    class_true_labels = np.array(true_labels)[class_indices]
    class_pred_labels = pred_labels[class_indices]
    
    # Calculate the accuracy for the specific class
    class_accuracy = accuracy_score(class_true_labels, class_pred_labels)
    return class_accuracy

def normalize_image(image):
    # Normalize the image to the range [0, 1]
    return (image - image.min()) / (image.max() - image.min())


def augment_underrepresented_classes(datas, label_counts):

    augmented_data = []
    # Exclude class 0 from the average count calculation
    avg_count = np.mean([count for label, count in label_counts.items() if label != 0])
    threshold = avg_count * 0.75  # Set threshold to 75% of the average count

    for label, count in label_counts.items():
        if count < threshold:
            samples_to_augment = [data for data in datas if data[1] == label]
            num_augmentations = int(threshold - count)
            for _ in range(num_augmentations):
                sample = random.choice(samples_to_augment)
                image = Image.fromarray(sample[0])
                
                # Randomly choose between the two augmentation functions
                if np.random.rand() > 0.5:
                    augmented_image = augmentation_fn(image)
                else:
                    augmented_image = augmentation_fn_combine(image, datas, label)
                
                augmented_data.append((np.array(augmented_image), label))

                # # Visualize original and augmented images
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # axes[0].imshow(image)
                # axes[0].set_title('Original Image')
                # axes[0].axis('off')
                # axes[1].imshow(augmented_image)
                # axes[1].set_title('Augmented Image')
                # axes[1].axis('off')
                # # plt.show()
                # # Save the plot
                # plt.savefig(f'./Augmented/augmented_image_{label}_{_}.png')
                # plt.close()

    datas.extend(augmented_data)
    return datas

def augmentation_fn(image):
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = ImageOps.mirror(image)
    
    
    # Random color enhancement
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(np.random.uniform(0.8, 1.2))
    
    # Random brightness enhancement
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(np.random.uniform(0.8, 1.2))
    
    # Random contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(np.random.uniform(0.8, 1.2))
    
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
    
def plot_samples(ds_val, predlabels, correct=True, data_name='Unknown', outfolder='./Data'):
    fig, ax = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(20,20))
    idx = -1
    for i in range(6):
        for j in range(6):
            while True:
                idx = np.random.choice(len(ds_val), 1, replace=False)
                if correct:
                    if ds_val[int(idx[0])]["labels"]> 0 and ds_val[int(idx[0])]["labels"] == int(predlabels[int(idx[0])]):
                        break
                else:
                    if ds_val[int(idx[0])]["labels"] != int(predlabels[int(idx[0])]):
                        break
 
            s = ds_val[int(idx[0])]
            image = np.transpose(s['images'], (1, 2, 0))
            image = normalize_image(image)  # Normalize the image
            ax[i,j].imshow(image)
            ax[i,j].set_title(f"G: {s['labels']}\nP: {int(predlabels[int(idx[0])])}")
            ax[i,j].axis('off')
    if correct:
        flag='correct'
    else:
        flag='wrong'
    plt.savefig(outfolder+'samples_'+data_name+'_'+flag+'.png')


def label_names():
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


def combine_datasets_in_batches(train_data, train_data_web, batch_size=5):
    combined_data = []
    web_index = 0
    web_len = len(train_data_web) - (len(train_data_web) % batch_size)  # Adjust web_len to discard residual

    for i in range(0, len(train_data), batch_size * 5):
        # Add a batch from the larger dataset
        combined_data.extend(train_data[i:i + batch_size * 5])

        # Add a batch from the smaller dataset
        if web_index < web_len:
            combined_data.extend(train_data_web[web_index:web_index + batch_size])
            web_index += batch_size

    return combined_data

def count_data_per_class_in_labels(labelss):
    label_counts = {label: 0 for label in label_names().keys()}
    label_namessss = {label: 0 for label in label_names().values()}

    class_names = label_names()

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

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def find_optimal_grid(n):
    """Return the optimal grid dimensions for plotting n images."""
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0 and i>1:
            return i, n // i
        elif n % i == 0 and i==1:
            return (n-1)//2, 2




def plot_samples_from_specific_label(ds, selected_predlabels, label_to_filter, data_name='Unknown', 
                                     outfolder='./Data'):
    
    label_name = label_names()[label_to_filter]
    if len(ds) < 36 and len(ds) > 6:
        idx = 0
        factors = find_optimal_grid(len(ds))
        grid1 = factors[0]
        grid2 = factors[1]
    elif len(ds) <= 6:
        idx = 0
        factors = (1, len(ds))
        grid1 = factors[0]
        grid2 = factors[1]
    else:
        idx = np.random.choice(len(ds)-36, 1, replace=False)[0]
        grid1 = 6
        grid2 = 6
    idx_original = idx
    channels_dict = {0: 'EL', 1: 'UV', 2: 'VIS'}
    channels = range(ds[0]['images'].shape[0])
    
    print(idx, channels, len(ds))
    for channel in channels:
        idx = idx_original
        fig, ax = plt.subplots(grid1, grid2, sharex=True, sharey=True, figsize=(20,20))
        if grid1 == 1 :
            for j in range(grid2):
                s = ds[int(idx)]
                image = np.transpose(s['images'][:3,:,:], (1, 2, 0))
                image = normalize_image(image)  # Normalize the image
                if image.shape[2]==3:
                    image = image[:,:,channel]
                ax[j].imshow(image, cmap='gray')
                if selected_predlabels is not None:
                    ax[j].set_title(f"Pred: {selected_predlabels[idx]}", fontsize=19)
                ax[j].axis('off')
                idx += 1
        else:
            for i in range(grid1):
                for j in range(grid2):
                   
                    s = ds[int(idx)]
                    image = np.transpose(s['images'][:3,:,:], (1, 2, 0))
                    image = normalize_image(image)  # Normalize the image
                    if image.shape[2]==3:
                        image = image[:,:,channel]
                    #image = normalize_image(image)  # Normalize the image
                    ax[i,j].imshow(image, cmap='gray')
                    if selected_predlabels is not None:
                        ax[i,j].set_title(f"Pred: {selected_predlabels[idx]}", fontsize=19)
                    ax[i,j].axis('off')
                    idx += 1
        channel_name = channels_dict[channel]
        plt.suptitle('Class:'+ label_name + ' ['+str(label_to_filter) + '], from '+data_name + ' in '+channel_name, fontsize=29)
        if os.path.exists(outfolder) == False:
            os.makedirs(outfolder)
        plt.savefig(outfolder+f'/samples_{data_name}_label_{label_name}_{channel_name}.png')

def find_last_checkpoint(output_dir):
    # List all checkpoint directories
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))  # Sort by checkpoint number

    # The last checkpoint will be the one with the highest step number
    last_checkpoint = checkpoints[-1] if checkpoints else None
    return last_checkpoint


def ploting_training_results(trainer, outfolder, last_checkpoint='', accuracies=[]):

    # Plot the loss function for training and evaluation data

    # Extract training and validation losses
    # Extract the log history
    if last_checkpoint:
        log_history = json.load(open(f'{outfolder}/{last_checkpoint}/trainer_state.json'))['log_history']
    else:
        log_history = trainer.state.log_history

    # Convert log history to DataFrame
    log_df = pd.DataFrame(log_history)

    # Extract training and validation losses
    # train_losses = log_df[log_df['loss'].notna()]['loss'].values
    val_acc= log_df[log_df['eval_accuracy'].notna()]['eval_accuracy'].values        

    # Plot the losses
    plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    if len(accuracies) ==2:
        plt.title(f'Validation Accuracy Duramat - {accuracies[0]:.2f} and Infinty {accuracies[1]:.2f}')
    else:
        plt.title('Validation Accuracy')
    plt.savefig(outfolder+'/loss_plot1.png')
    plt.close()

def plot_samples_from_all_labels_with_acc(ds_val, predlabels, accuracy, class_accuracies, correct=True, data_name='Unknown', outfolder='./Data'):
    unique_labels = np.unique([s['labels'] for s in ds_val])
    for label in unique_labels:
        if class_accuracies.get(label) == 0.:
            break
        if class_accuracies.get(label) < 0.6:
            plot_samples_from_specific_label_with_acc(ds_val, predlabels, label, accuracy, class_accuracies, False, data_name, outfolder)
        else:
            plot_samples_from_specific_label_with_acc(ds_val, predlabels, label, accuracy, class_accuracies, correct, data_name, outfolder)

def plot_samples_from_specific_label_with_acc(ds_val, predlabels, label_to_filter, accuracy, class_accuracies, correct=True, data_name='Unknown', outfolder='./Data'):
    fig, ax = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(20,20))
    idx = -1
    for i in range(6):
        for j in range(6):
            while True:
                idx = np.random.choice(len(ds_val), 1, replace=False)
                if ds_val[int(idx[0])]["labels"] == label_to_filter:
                    if correct:
                        if ds_val[int(idx[0])]["labels"] == int(predlabels[int(idx[0])]):
                            break
                    else:
                        if ds_val[int(idx[0])]["labels"] != int(predlabels[int(idx[0])]):
                            break
 
            s = ds_val[int(idx[0])]
            image = np.transpose(s['images'], (1, 2, 0))
            image = normalize_image(image)  # Normalize the image
            ax[i,j].imshow(image)
            ax[i,j].set_title(f"G: {s['labels']}\nP: {int(predlabels[int(idx[0])])}", fontsize=19)
            ax[i,j].axis('off')
    if correct:
        flag='correct'
    else:
        flag='wrong'
    plt.suptitle(f'Total Accuracy of {data_name} is: {accuracy:.3f}. Class Accuracy: {class_accuracies.get(label_to_filter, 0):.3f}', fontsize=30)

    plt.savefig(outfolder+f'/samples_{data_name}_label_{str(label_to_filter)}_{flag}.png')

