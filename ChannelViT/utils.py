from sklearn.metrics import f1_score, accuracy_score, log_loss
import torch
from transformers import  EvalPrediction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import json
from PIL import Image, ImageEnhance, ImageOps

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

def plot_samples_from_all_labels(ds_val, predlabels, accuracy, class_accuracies, correct=True, data_name='Unknown', outfolder='./Data'):
    unique_labels = np.unique([s['labels'] for s in ds_val])
    for label in unique_labels:
        if class_accuracies.get(label) == 0.:
            break
        if class_accuracies.get(label) < 0.6:
            plot_samples_from_specific_label(ds_val, predlabels, label, accuracy, class_accuracies, False, data_name, outfolder)
        else:
            plot_samples_from_specific_label(ds_val, predlabels, label, accuracy, class_accuracies, correct, data_name, outfolder)

def plot_samples_from_specific_label(ds_val, predlabels, label_to_filter, accuracy, class_accuracies, correct=True, data_name='Unknown', outfolder='./Data'):
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