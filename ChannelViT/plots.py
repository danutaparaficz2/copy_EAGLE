import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
from utils import label_names, normalize_image, find_optimal_grid, select_images_by_label




def save_images_by_label(images, labelss, output_dir, flag='', name_flag='', original_indices=None):
    """
    Save images into separate folders based on their labels, enhancing contrast for better visibility.

    Args:
        images (list): List of image arrays (NumPy arrays).
        labels (list): List of labels corresponding to the images.
        output_dir (str): Path to the output directory where folders will be created.
    """
    if original_indices is None:
        original_indices = range(len(images))
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, (image, labels) in enumerate(zip(images, labelss)):
        # Use only the first label if there are multiple labels
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            if len(labels) > 0:
                labels = labels
            else:
                continue
        else:
            labels = [labels]
        for label in labels:    
            imag = image[0]
            # Create a folder for the label
            label_folder = os.path.join(output_dir, label_names(flag=flag)[label])
            os.makedirs(label_folder, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Enhance contrast and save the image
            if imag.ndim == 3:
                if name_flag == 'gray':
                    # Split the RGB image into its channels
                    red_channel = imag[:, :, 0]
                    green_channel = imag[:, :, 1]
                    blue_channel = imag[:, :, 2]

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

                    # Save the combined image
                    combined_image.save(os.path.join(label_folder, f'image_{i}_gray_channels.png'))
                elif name_flag == 'rgb':
                    if type(imag) == torch.Tensor:
                        imag = np.transpose(imag, (1, 2, 0)).numpy()
                    # Split the RGB image into its channels
                    red_channel = imag[:, :, 0]
                    green_channel = imag[:, :, 1:4]
                    blue_channel = imag[:, :, 4:7]

                    # Normalize and enhance contrast for each channel

                    red_channel = (red_channel - np.min(red_channel)) / (np.max(red_channel) - np.min(red_channel) + 1e-8) * 255
                    green_channel = (green_channel - np.min(green_channel)) / (np.max(green_channel) - np.min(green_channel) + 1e-8) * 255
                    blue_channel = (blue_channel - np.min(blue_channel)) / (np.max(blue_channel) - np.min(blue_channel) + 1e-8) * 255

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

                    # Save the combined image (original size)
                    combined_image.save(os.path.join(label_folder, f'image_{i}_rbg_channels'+str(original_indices[i])+'.png'))      

                    # # Also save a resized version (224x224)
                    # combined_image_224 = combined_image.resize((672, 224))
                    # combined_image_224.save(os.path.join(label_folder, f'image_{i}_rbg_channels_224.png'))
            elif imag.ndim == 2:
                # Normalize and enhance contrast for grayscale images
                #imag = normalize_image(imag) * 255
                # Normalize image to [0, 255] for better contrast
                norm_image = (imag - np.min(imag)) / (np.max(imag) - np.min(imag) + 1e-8) * 255
                norm_image = norm_image.astype(np.uint8)
                image_pil = Image.fromarray(norm_image)
                image_pil.save(os.path.join(label_folder, f'image_{i}'+str(original_indices[i])+'.png'))
                # image_224.save(os.path.join(label_folder, f'image_{i}_224.png'))


    print(f"Images saved in folders under {output_dir}")

    
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

# def find_last_checkpoint(output_dir):
#     # List all checkpoint directories
#     checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
#     checkpoints.sort(key=lambda x: int(x.split('-')[1]))  # Sort by checkpoint number

#     # The last checkpoint will be the one with the highest step number
#     last_checkpoint = checkpoints[-1] if checkpoints else None
#     return last_checkpoint


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
    # ensure output directory exists
    try:
        os.makedirs(outfolder, exist_ok=True)
    except Exception:
        pass
    plt.savefig(os.path.join(outfolder, 'loss_plot1.png'))
    plt.close()

def plot_samples_from_all_labels_with_acc(ds_val, predlabels, class_accuracies, data_name='Unknown', outfolder='./Data', certainty= None):
    unique_labels = list(class_accuracies.keys())
    for label in unique_labels:
        if class_accuracies.get(label) == 0.:
            break
        selected_images, selected_predlabels, selected_certainty = select_images_by_label(ds_val, predlabels, label, certainty=certainty) # selects images that originally are labeled in specific label


        if class_accuracies.get(label) < 0.9:
            plot_samples_from_specific_label_with_acc(selected_images, selected_predlabels, label, class_accuracies, False, data_name, outfolder, certainty=selected_certainty)
            plot_samples_from_specific_label_with_acc(selected_images, selected_predlabels, label, class_accuracies, True, data_name, outfolder, certainty=selected_certainty)

        else:
            plot_samples_from_specific_label_with_acc(selected_images, selected_predlabels, label, class_accuracies, True, data_name, outfolder, certainty=selected_certainty)
            plot_samples_from_specific_label_with_acc(selected_images, selected_predlabels, label, class_accuracies, False, data_name, outfolder, certainty=selected_certainty)

def find_agreement_indices(true_labels, predlabels):
    """
    Find indices where predictions agree/disagree with true labels
    
    Args:
        true_labels: List of one-hot encoded tensors
        predlabels: List of lists with predicted label indices
    
    Returns:
        matching_indices: List of indices where predictions match
        non_matching_indices: List of indices where predictions don't match
    """
    matching_indices = []
    non_matching_indices = []
    
    for i, (true_tensor, pred) in enumerate(zip(true_labels, predlabels)):
        # Convert one-hot tensor to list of indices where value is 1
        true_indices = torch.where(true_tensor == 1)[0].tolist()
        
        # Convert to sets for comparison
        true_set = set(true_indices)
        pred_set = set(pred)
        
        # Check if they match exactly
        if true_set == pred_set:
            matching_indices.append(i)
        else:
            non_matching_indices.append(i)
    
    return matching_indices, non_matching_indices


def plot_samples_from_specific_label_with_acc(ds_val, predlabels, label_to_filter, class_accuracies, correct=True, data_name='Unknown', outfolder='./Data', certainty=None):
    # empty_indices = [i for i, x in enumerate(predlabels) if not x]
    # predlabels = [x for i, x in enumerate(predlabels) if i not in empty_indices]
    # ds_val = [x for i, x in enumerate(ds_val) if i not in empty_indices]
    # certainty = [x for i, x in enumerate(certainty) if i not in empty_indices]

    non_matching_indices = []
    matching_indices = []
    for i, pred in enumerate(predlabels):
        if label_to_filter in pred:
            matching_indices.append(i)
        else:
            non_matching_indices.append(i)

    print(matching_indices, non_matching_indices)
    if correct:
        if len(matching_indices) > 36:
            indices_to_use = np.random.choice(matching_indices, 36, replace=False)
        else:
            indices_to_use = matching_indices
    else:
        if len(non_matching_indices) > 36:
            indices_to_use = np.random.choice(non_matching_indices, 36, replace=False)
        else:
            indices_to_use = non_matching_indices

    num_images = min(len(indices_to_use), 36)
    if num_images == 0:
        print(f"No samples to plot for label {label_to_filter} (correct={correct}).")
        return

    grid1, grid2 = find_optimal_grid(num_images)
    # If only one image, make sure ax is always 2D for consistent indexing
    fig, ax = plt.subplots(grid1, grid2, sharex=True, sharey=True, figsize=(20,20))
    ax = np.array(ax)
    ax = ax.reshape(-1)  # Flatten for easy indexing


    for idx, index in enumerate(indices_to_use):
        if idx >= len(ax):
            break  # Prevent IndexError if more images than subplots
        s = ds_val[index]
        if s['images'].shape[0] != 1:
            image = np.transpose(s['images'][0:1,:,:], (1, 2, 0))
        else:
            image = np.transpose(s['images'], (1, 2, 0))
        # Scale image from mean~0, std~1 to [0, 1] for display
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        ax[idx].imshow(image, cmap='gray')
        if certainty is not None:
            certainty_value = certainty[index][predlabels[index][0]]
            certainty_value_for_ground_truth = certainty[index][label_to_filter]
            ax[idx].set_title(f"G: {label_to_filter} ({certainty_value_for_ground_truth:.2f})\nP: {predlabels[index]} ({certainty_value:.2f})", fontsize=19)
        else:
            ax[idx].set_title(f"G: {label_to_filter}\nP: {predlabels[index]}", fontsize=19)
        ax[idx].axis('off')

    # Hide any unused subplots
    for idx in range(len(indices_to_use), len(ax)):
        ax[idx].axis('off')

    if correct:
        flag='correct'
    else:
        flag='wrong'
    plt.suptitle(f'Data from {data_name}. Class Accuracy: {class_accuracies.get(label_to_filter, 0):.3f}', fontsize=30)

    plt.savefig(outfolder+f'/samples_{data_name}_label_{str(label_to_filter)}_{flag}.png')



def plot_multilabel_confusion_matrix(true_labels, predicted_labels, class_names, output_path=None):
    """
    Plot multilabel confusion matrices for each class.
    """
    mcm = multilabel_confusion_matrix(true_labels, predicted_labels)
    print(mcm)
    for i, (cm, class_name) in enumerate(zip(mcm, class_names)):
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'not {class_name}', class_name])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix for class: {class_name}')
        if output_path:
            plt.savefig(f"{output_path}_class_{class_name}.png")
        plt.close()
    return mcm

def plot_normalized_confusion_matrix(true_labels, predicted_labels, class_names, output_path=None):
    """
    Calculate and plot the normalized confusion matrix (values as percentages).

    Args:
        true_labels (np.ndarray): Array of true labels (integer labels).
        predicted_labels (np.ndarray): Array of predicted labels (integer labels).
        class_names (list): List of class names corresponding to the class indices.
        output_path (str): Path to save the confusion matrix plot (optional).

    Returns:
        normalized_cm (np.ndarray): Normalized confusion matrix as a NumPy array.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Normalize the confusion matrix by row (percentage of each class)
    normalized_cm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(normalized_cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)

    # Annotate the matrix with percentage values
    for i in range(normalized_cm.shape[0]):
        for j in range(normalized_cm.shape[1]):
            plt.text(j, i, f"{normalized_cm[i, j]:.1f}%",  # Format as percentage
                     ha="center", va="center",
                     color="white" if normalized_cm[i, j] > 50 else "black")  # Adjust text color for visibility

    plt.tight_layout()

    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path)

    return normalized_cm

def confusion_matrix_per_class(pred_labels, true_labels, plot=False, normalize=False):
    # For each class, print and plot the confusion matrix
    cm = {}
    for class_idx in range(true_labels.shape[1]):
        y_true = true_labels[:, class_idx]
        y_pred = pred_labels[:, class_idx]
        cm[class_idx] = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        if normalize:
            title = f"Normalized confusion matrix for class {class_idx}"
        else:
            title = f"Confusion matrix for class {class_idx}"
 
        print(title)
        print( [class_idx])
        if plot:
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm[class_idx])
            disp.plot()
            plt.title(title)
            plt.savefig(f"confusion_matrix_class_{class_idx}.png")
    return cm


# def plot_samples_from_all_labels(ds, predlabels, unique_labels, data_name='Unknown', outfolder='./Data'):


#     for label in unique_labels:
#         selected_images, selected_predlabels = select_images_by_label(ds, predlabels, label)
#         plot_samples_from_specific_label(selected_images, selected_predlabels, label, data_name, outfolder)


# def save_data(current_dir, data_Duramat):
#     # Save data_Duramat in a format compatible with torchvision.datasets.ImageFolder
#     output_dir = current_dir + '/Data/Duramat_ImageFolder/'
#     os.makedirs(output_dir, exist_ok=True)

#     for idx, (image_np, label) in enumerate(data_Duramat):
#         # Convert one-hot label to integer
#         label_idx = torch.where(label == 1)[0][0].item()
#         label_dir = os.path.join(output_dir, str(label_idx))
#         os.makedirs(label_dir, exist_ok=True)
        
#         # Save the image as a .png file
#         image_path = os.path.join(label_dir, f'image_{idx}.png')
#         Image.fromarray(image_np).save(image_path)



# def plot_samples(ds_val, predlabels, correct=True, data_name='Unknown', outfolder='./Data'):
#     fig, ax = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(20,20))
#     idx = -1
#     for i in range(6):
#         for j in range(6):
#             while True:
#                 idx = np.random.choice(len(ds_val), 1, replace=False)
#                 if correct:
#                     if ds_val[int(idx[0])]["labels"]> 0 and ds_val[int(idx[0])]["labels"] == int(predlabels[int(idx[0])]):
#                         break
#                 else:
#                     if ds_val[int(idx[0])]["labels"] != int(predlabels[int(idx[0])]):
#                         break
 
#             s = ds_val[int(idx[0])]
#             image = np.transpose(s['images'], (1, 2, 0))
#             image = normalize_image(image)  # Normalize the image
#             ax[i,j].imshow(image)
#             ax[i,j].set_title(f"G: {s['labels']}\nP: {int(predlabels[int(idx[0])])}")
#             ax[i,j].axis('off')
#     if correct:
#         flag='correct'
#     else:
#         flag='wrong'
#     plt.savefig(outfolder+'samples_'+data_name+'_'+flag+'.png')


# def combine_datasets_in_batches(train_data, train_data_web, batch_size=5):
#     combined_data = []
#     web_index = 0
#     web_len = len(train_data_web) - (len(train_data_web) % batch_size)  # Adjust web_len to discard residual

#     for i in range(0, len(train_data), batch_size * 5):
#         # Add a batch from the larger dataset
#         combined_data.extend(train_data[i:i + batch_size * 5])

#         # Add a batch from the smaller dataset
#         if web_index < web_len:
#             combined_data.extend(train_data_web[web_index:web_index + batch_size])
#             web_index += batch_size

#     return combined_data