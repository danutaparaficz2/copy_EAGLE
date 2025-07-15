from importlib_metadata import files
import torch
import re
import pandas as pd
from PIL import Image
import numpy as np
import sys
import os
import importlib
from torchvision import transforms
from utils import label_names, normalize_image

from collections import defaultdict
import os


import torch
from torchvision import transforms
from tqdm import tqdm


def just_transform(data, channels=[0]):
    """
    Preprocess all images (resize, normalize, stack channels) and save as a single .pt file.
    """

    images = [item[0] for item in data]  # Extract images
    labels = [item[1] for item in data]  # Extract labels
    # Define transforms
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
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.73], std=[0.17])  # Normalize the image
 ])

    tensors = []
    for img in tqdm(images, desc="Preprocessing images"):
        if img.shape[-1] == 7:
            el = img[:, :, 0]
            vis = img[:, :, 1:4]
            uv = img[:, :, 4:7]
            el = el_transform(el)
            vis = rgb_transform(vis)
            uv = rgb_transform(uv)
            img_chw = torch.cat([el, vis, uv], dim=0)  # (7, 224, 224)
        else:
            if len(channels) == 1:
                img_chw = transform(img)
            else:
                img_chw = el_transform(img)

        tensors.append(img_chw)

    # make list out of tensors and labels
    tensor_label_list = list(zip(tensors, labels))

    # dataset_all = PVDataset(tensor_label_list, channels=channels,  scale=1, return_labels=return_labels)

    return tensor_label_list


def normalize_image_0_255(files_by_folder, PATH, tech, this_folders_only=[], folders_excluded=[], all_colors=False):
    normalized_images = []
    filtered_labels = []
    for folder, files in files_by_folder.items():
        if len(folders_excluded) != 0:
            if any(f in folder for f in folders_excluded):
                # Display and save images in the folder
                for filename, label in files:
                    # Allow for one-hot encoding with multiple ones (multi-label)
                    if isinstance(label, (np.ndarray, list)):
                        label_indices = [i for i, v in enumerate(label) if v == 1]
                        if not label_indices:
                            continue
                    else:
                        label_indices = [str(label)]

                    image_path = os.path.join(PATH, 'segments', filename)
                    if tech == '_EL_':
                        try:
                            image = Image.open(image_path).convert('L')
                            # Find corresponding UV and VI images
                            uv_filename = filename.replace('_EL_', '_UV_')
                            vi_filename = filename.replace('_EL_', '_VI_')
                            uv_path = os.path.join(PATH, 'segments', uv_filename)
                            vi_path = os.path.join(PATH, 'segments', vi_filename)
                            try:
                                image_uv = Image.open(uv_path).convert('L')
                                image_vi = Image.open(vi_path).convert('L')
                            except Exception as e:
                                print(f"Could not load UV or VI image for {filename}: {e}")
                                continue

                            import matplotlib.pyplot as plt

                            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                            axs[0].imshow((image), cmap='gray')
                            axs[0].set_title('EL')
                            axs[1].imshow(255-np.array(image_uv), cmap='gray')
                            axs[1].set_title('UV ')
                            axs[2].imshow(255-np.array(image_vi), cmap='gray')
                            axs[2].set_title('VI')
                            for ax in axs:
                                ax.axis('off')
                            save_folder = os.path.join('Data/images/', folder)
                            os.makedirs(save_folder, exist_ok=True)
                            file_only = os.path.basename(filename.replace('.tif', '.png'))
                            for label_idx in label_indices:
                                label_folder = os.path.join(save_folder, label_names(flag='Website')[label_idx])
                                os.makedirs(label_folder, exist_ok=True)
                                save_path = os.path.join(label_folder, file_only)
                                plt.tight_layout()
                                # Combine all label indices for this image into a string
                                label_names(flag='Website')
                                label_indices_str = ",".join(label_names(flag='Website')[idx] for idx in label_indices)
                                plt.title(f'Labels: {label_indices_str}')
                                plt.savefig(save_path)

                            plt.close()
                        except Exception as e:
                            print(f"Error displaying or saving {image_path}: {e}")
                continue 
        if len(this_folders_only) != 0:
            if any(f in folder for f in this_folders_only):
                print(folder)

                updated_filenames  = [filename.replace('_EL_', tech) for filename, label in files]
                mean, std, max, min = calculate_mean_std_per_folder(PATH+'/segments/', updated_filenames)
                print(f"Mean: {mean}, Std: {std}, Max: {max}, Min: {min}")
                normalized_images.extend(load_and_normalize_images(PATH+'/segments/', updated_filenames, mean, std, max, min, all_colors=all_colors))
                labels = [label for _, label in files]
                filtered_labels.extend(labels)

        else:   
            print(folder)

            updated_filenames = [filename.replace('_EL_', tech) for filename, label in files]
            labels = [label for _, label in files]
            # Find indices where filenames are duplicated
            filename_indices = defaultdict(list)
            for idx, fname in enumerate(updated_filenames):
                filename_indices[fname].append(idx)
            duplicate_indices = {fname: idxs for fname, idxs in filename_indices.items() if len(idxs) > 1}
            if duplicate_indices:
                print(f"Duplicate filename indices in folder '{folder}' : {duplicate_indices}")
                # Remove the first occurrence of each duplicate filename from updated_filenames
                for fname, idxs in duplicate_indices.items():
                    if idxs:
                        # Remove the first occurrence
                        updated_filenames[idxs[1]] = None
                        labels[idxs[1]] = None  # Also remove the corresponding label
                # Remove all None entries from updated_filenames and corresponding labels
                updated_filenames = [f for f in updated_filenames if f is not None]
                labels = [l for l in labels if l is not None]
            # Step 1: Calculate mean and std for the folder
            mean, std, max, min = calculate_mean_std_per_folder(PATH+'/segments/', updated_filenames)
            print(f"Mean: {mean}, Std: {std}", f"Max: {max}, Min: {min}")

            # Step 2: Load and normalize images using the calculated mean and std
            normalized_images.extend(load_and_normalize_images(PATH+'/segments/', updated_filenames, mean, std, max, min, all_colors=all_colors))

            filtered_labels.extend(labels)
            # print(np.min([image.min() for image in normalized_images]))
            # print(np.max([image.max() for image in normalized_images]))
            # print(f"Loaded and normalized {len(normalized_images)} images.")
    return normalized_images, filtered_labels

def calculate_mean_std_per_folder(folder_path, image_files):
    """
    Calculate the mean and standard deviation of images in a folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        mean (float): Mean pixel value of all images in the folder.
        std (float): Standard deviation of pixel values of all images in the folder.
    """
    pixel_values = []

    for image_file in image_files:
        image = np.array(Image.open(folder_path+image_file).convert('L')).astype(np.float32)  # Convert to grayscale
        pixel_values.extend(list(image.flatten()))


    # Calculate mean and std
    mean = np.mean(pixel_values)
    std = np.std(pixel_values)
    max = np.max(pixel_values)
    min = np.min(pixel_values)
    return mean, std, max, min

def load_and_normalize_images(folder_path, image_files, mean, std, max, min, all_colors=False):
    """
    Load and normalize images from a folder using the given mean and std.

    Args:
        folder_path (str): Path to the folder containing images.
        mean (float): Mean pixel value for normalization.
        std (float): Standard deviation of pixel values for normalization.

    Returns:
        normalized_images (list): List of normalized images as NumPy arrays.
    """
    normalized_images = []

    for image_file in image_files:
        if all_colors:
            # Load the image as RGB
            image = np.array(Image.open(folder_path+image_file).convert('RGB')).astype(np.float32)
        else:
            image = np.array(Image.open(folder_path+image_file).convert('L')).astype(np.float32)  # Convert to grayscale
        # Normalize the image
        # normalized_image = ((image - mean) / std) * 255.
        normalized_image = (image - min) / (max - min) * 255
        # Scale back to 0-255 range
        # normalized_image = (normalized_image - normalized_image.min()) / (normalized_image.max() - normalized_image.min()) * 255
        normalized_images.append(normalized_image.astype(np.uint8))
        # Save the normalized images as PNG files
        os.makedirs(folder_path+'/normalized/', exist_ok=True)
        Image.fromarray(normalized_image.astype(np.uint8)).save(folder_path+'/normalized/'+os.path.basename(image_file).replace('.tif', '_normalized.png'))
    return normalized_images

def separate_files_by_folder(file_paths, labels=None):
    """
    Separate file paths into groups based on their parent folders.

    Args:
        file_paths (list): List of file paths.

    Returns:
        dict: A dictionary where keys are folder names and values are lists of file paths.
    """
    folder_dict = defaultdict(list)

    for idx, file_path in enumerate(file_paths):
        # Extract the folder name (parent directory)
        folder_name = os.path.dirname(file_path)
        if labels:
            folder_dict[folder_name].append((file_path, labels[idx]))
        else:
            folder_dict[folder_name].append((file_path, 'None'))

    return folder_dict


def save_image_from_array(image_array, file_path):
    # Ensure the array is in the correct format (uint8)
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    # Create an image from the array
    image = Image.fromarray(image_array)
    
    # Save the image
    image.save(file_path)
def stack_images(images_list):
    stacked_images = torch.cat([torch.tensor(image) for image in images_list], dim=0)
    return stacked_images

class Load_Data:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_pickle(self.path)
        self.df = self.df[self.df.labels >= 0]
        self.labels = self.df.labels.unique()
        self.labels_as_integers = [int(label) for label in self.df['labels'].values]
        self.images = [image.astype(np.float32) for image in list(self.df.images.values)]
        for image in self.images:
            if np.any(image > 255) or np.any(image < 0):
                raise ValueError("Image values should be in the range [0, 255]")
        self.images = [image.astype(np.uint8) for image in self.images]
        self.images = [np.array(Image.fromarray(image).convert('L')) for image in self.images]
        self.data = list(zip(self.images, self.labels_as_integers))


    def get_data(self):
        return self.data
    def get_just_images(self):
        return self.images
    def get_label_statistics(self):
        label_counts = {}
        for label in self.labels_as_integers:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        label_counts = dict(sorted(label_counts.items()))
        print(label_counts)
        label_counts_named = {(label_names())[int_label]: count for int_label, count in label_counts.items()}
        print(label_counts_named)
        return label_counts
    
class Load_Data_Handler:
    def __init__(self, PATH, args, classified_by=["Ebrar", 'Ralf'],  this_folders_only=[], folders_excluded=[]):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(current_dir, '../eagle-jsonhandler'))
        sys.path.append(os.path.join(current_dir, '../'))
        eagle_jsonhandler = importlib.import_module("JSONHandler")
        
        panellist = eagle_jsonhandler.getGroupedPanelList(PATH+'/overviews/')
        classified_cells = []
        for classified_by_one in classified_by:
            classified_cells.extend(eagle_jsonhandler.getCellsByAttribute(groupedPanels=panellist, attribute="classifiedBy", values=classified_by_one))

        elpaths, uvpaths, vispaths, labels = eagle_jsonhandler.getCellsImagePathsAndLabels(classified_cells)
        # label_types = ["good", "crack", "cross", "dark", "corrosion"]
        # label_counts = np.sum(labels, axis=0)


        # Separate files by their folders
        files_by_folder = separate_files_by_folder(elpaths, labels)
        if args.use_only_EL:

            self.images_el, self.labels = normalize_image_0_255(files_by_folder, PATH, '_EL_', this_folders_only=this_folders_only, folders_excluded=folders_excluded)
            self.images =self.images_el
        else:
            for technology in ['_EL_', '_UV_', '_VI_']:
                print(f"Processing technology: {technology}")
                if technology == '_EL_':
                    
                    self.images_el, self.labels = normalize_image_0_255(files_by_folder, PATH, technology, this_folders_only=this_folders_only, folders_excluded=folders_excluded)

                elif technology == '_UV_':
                    self.images_uv, self.labels = normalize_image_0_255(files_by_folder, PATH, technology, this_folders_only=this_folders_only, folders_excluded=folders_excluded, all_colors=args.all_colors)
                elif technology == '_VI_':
                    self.images_vis, self.labels = normalize_image_0_255(files_by_folder, PATH, technology, this_folders_only=this_folders_only, folders_excluded=folders_excluded, all_colors=args.all_colors)
            
            # Combine the 3 lists of grayscale images into one list of RGB images
            # self.images = [np.stack((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
            # self.images = [np.concatenate((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
            if args.all_colors:
                self.images = [np.concatenate((el[:,:,None], uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
            else:
                self.images = [np.stack((el, uv, vis), axis=-1)for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]

        self.labels_as_integers = [np.where(label == 1)[0].tolist() for label in self.labels]
        self.empty_label_indices = [i for i, label in enumerate(self.labels) if isinstance(label, np.ndarray) and np.all(label == 0)]

        # Remove labels at indices in self.empty_label_indices
        filtered_images = [img for i, img in enumerate(self.images) if i not in self.empty_label_indices]
        filtered_labels = [torch.tensor(label) for i, label in enumerate(self.labels) if i not in self.empty_label_indices]
        self.data = list(zip(filtered_images, filtered_labels))
        

    def get_data(self):
        return self.data
    
    def get_label_statistics(self):
        label_counts = {}
        for label in self.labels_as_integers:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        return label_counts

class Load_Data_Handler_notlabeled:
    def __init__(self, PATH, subfolder):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        panelfolder = PATH+'/segments/'+subfolder
        elpaths =  [file for file in os.listdir(panelfolder) if "_EL_" in file and file.endswith(".tif")]
        uvpaths =  [file for file in os.listdir(panelfolder) if "_UV_" in file and file.endswith(".tif")]
        vispaths = [file for file in os.listdir(panelfolder) if "_VI_" in file and file.endswith(".tif")]
        
        label_types = ["good", "crack", "cross", "dark", "corrosion"]
        
        # sorted_elpaths = sorted(elpaths, key=lambda x: (int(re.search(r'^(\d+)', x).group(1)), int(re.search(r'Cell(\d+)', x).group(1))))
        # sorted_uvpaths = sorted(uvpaths, key=lambda x: (int(re.search(r'^(\d+)', x).group(1)), int(re.search(r'Cell(\d+)', x).group(1))))
        # sorted_vispaths = sorted(vispaths, key=lambda x: (int(re.search(r'^(\d+)', x).group(1)), int(re.search(r'Cell(\d+)', x).group(1))))
        elpaths.sort()
        uvpaths.sort()
        vispaths.sort()
        assert len(elpaths) == len(uvpaths) == len(vispaths), "Mismatch in number of images across channels" 
        elpaths_with_subfolder = [os.path.join(subfolder, fname) for fname in elpaths]

        files_by_folder = separate_files_by_folder(elpaths_with_subfolder)

        for technology in ['_EL_', '_UV_', '_VI_']:
            if technology == '_EL_':
                self.images_el = normalize_image_0_255(files_by_folder, PATH, technology)
            elif technology == '_UV_':
                self.images_uv = normalize_image_0_255(files_by_folder, PATH, technology)
            elif technology == '_VI_':
                self.images_vis = normalize_image_0_255(files_by_folder, PATH, technology)
        # self.images_el = [np.array(Image.open(PATH+'/segments/'+subfolder+'/'+path).convert('L')).astype(np.uint8) for path in elpaths]
        # self.images_uv = [np.array(Image.open(PATH+'/segments/'+subfolder+'/'+path).convert('L')).astype(np.uint8) for path in uvpaths]
        # self.images_vis = [np.array(Image.open(PATH+'/segments/'+subfolder+'/'+path).convert('L')).astype(np.uint8) for path in vispaths]

        # Combine the 3 lists of grayscale images into one list of RGB images
        # self.images = [np.stack((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        # self.images = [np.concatenate((el, uv, vis), axis=-1) for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.images = [np.stack((el, uv, vis), axis=-1)for el, uv, vis in zip(self.images_el, self.images_uv, self.images_vis)]
        self.data = list(zip(self.images))
        self.sorted_elpaths = elpaths
        
    def get_data(self):
        return self.data, self.sorted_elpaths
    


class PVDataset(torch.utils.data.Dataset):
    def __init__(self, df, channels, scale=1, return_labels=True):
        self.df = df
        self.channels = channels

        self.scale = scale
        self.return_labels = return_labels


    def __getitem__(self, idx):
        row = self.df[idx]
        img_chw = row[0]

        # Select the specified channels
        img_chw = img_chw[self.channels[idx], :, :]

        # Scale if needed
        if self.scale != 1:
            img_chw *= self.scale

        self.channel = torch.tensor([c for c in self.channels[idx]])

        if self.return_labels:
            return {"images": img_chw, "labels": row[1], "channels": self.channel}
        else:
            return {"images": img_chw, "channels": self.channel}
        
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)


def load_data_from_csv(csv_path):
    """
    Load data from a CSV file. Assumes the CSV contains 'image_path' and 'label' columns.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        list: A list of tuples where each tuple contains an image array and its corresponding label.
    """
    data = []
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        image_path = row['image_path']
        label = row['label']

        # Load the image and convert it to grayscale
        image = np.array(Image.open(image_path).convert('L')).astype(np.uint8)

        data.append((image, label))

    return data

import torch
from torch.utils.data import Sampler, ConcatDataset, DataLoader

class AlternatingBatchSampler(Sampler):
    def __init__(self, len1, len2, batch_size):
        self.len1 = len1
        self.len2 = len2
        self.batch_size = batch_size
        self.indices1 = list(range(self.len1))
        self.indices2 = list(range(self.len2))
        self.ptr1 = 0
        self.ptr2 = 0

    def __iter__(self):
        ptr1, ptr2 = 0, 0
        while ptr1 < self.len1 or ptr2 < self.len2:
            if ptr1 < self.len1:
                yield self.indices1[ptr1:ptr1+self.batch_size]
                ptr1 += self.batch_size
            if ptr2 < self.len2:
                # Offset indices for the second dataset in ConcatDataset
                yield [i + self.len1 for i in self.indices2[ptr2:ptr2+self.batch_size]]
                ptr2 += self.batch_size

    def __len__(self):
        return ((self.len1 + self.batch_size - 1) // self.batch_size) + ((self.len2 + self.batch_size - 1) // self.batch_size)