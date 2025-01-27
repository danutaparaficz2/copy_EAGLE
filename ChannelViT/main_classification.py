import torch
import os
import numpy as np
import re
import argparse
PYTORCH_ENABLE_MPS_FALLBACK=1
#### Local imports
from load_data import Load_Data, PVDataset
from utils import  plot_samples, ploting_training_results, plot_samples_from_all_labels, find_last_checkpoint, calculate_class_accuracy
from training import  init_trainer, load_model, data_split_and_transform, train_save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training and evaluation.')
    parser.add_argument('--in_chans', type=int, default=18, help='Number of input channels.')
    parser.add_argument('--init_weights_name', type=str, default='so2sat_channelvit_small_p8_with_hcs_hard_split_supervised', help='Name of the initial weights file.')
   # imagenet_channelvit_small_p16_with_hcs_supervised, so2sat_channelvit_small_p8_with_hcs_hard_split_supervised
    args = parser.parse_args()

    def extract_number_from_name(name):
        match = re.search(r'p(\d+)', name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("No number found in the name")

    args.patch_size = extract_number_from_name(args.init_weights_name)
    return args

args = parse_args()


if __name__ == '__main__':

    ######################################### Set the parameters ##################################################

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    output_folder = '/Data_augmented/model_with_'+args.init_weights_name+'/epochs_'+str(args.num_train_epochs)+'/'

    # Find the last checkpoint
    last_checkpoint = find_last_checkpoint(current_dir+output_folder)

    print(f"Last checkpoint: {last_checkpoint}")
    ######################################### Load the data ##################################################

    path = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
    data_loader =  Load_Data(path)
    datas = data_loader.get_data()
    label_counts = data_loader.get_label_statistics()
    train_dataset, val_dataset, transform = data_split_and_transform(datas)
    
    # PATH_DATA = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    # data_loader_2 = Load_Data_Handler(PATH_DATA)
    # data = data_loader_2.get_data()
    
    #################################################### ONLY TRAINING MODE  #########################################################
    # Model with originally pretrained weights
    model = load_model(args, current_dir+'/Data/', device, args.init_weights_name)
    trainer = init_trainer(args, model, val_dataset, current_dir+output_folder)
    trainer = train_save_model(trainer, train_dataset, val_dataset, current_dir+output_folder)
    ploting_training_results(trainer, current_dir+output_folder)

    ########################################### ONLY PREDICT ###########################################
    ########## Duramat ##########
    # Load the trained model back into the trainer
    model = load_model(args, current_dir+output_folder, device, 'trained_state_dict')

    #This method is used to set the model to evaluation mode. It is important to call this method before running inference, 
    # because the model needs to know that it is in evaluation mode so that it can turn off features like dropout and batch normalization.
    model.eval()
    trainer = init_trainer(args, model, val_dataset, current_dir+output_folder)
    # Use the trainer to make predictions
    predictions = trainer.predict(val_dataset)
    predlabels = predictions.predictions.argmax(axis=-1)
    # Extract the true labels from val_dataset
    true_labels = np.array([label for _, label in val_dataset.df])
    print(predictions.metrics)
    accuracy_Duramat = predictions.metrics['test_accuracy']
    # Calculate accuracy for each class
    unique_labels = np.unique(true_labels)
    class_accuracies = {}
    for label in unique_labels:
        class_accuracies[label] = calculate_class_accuracy(true_labels, predlabels, class_label=label)

    print("Class accuracies:", class_accuracies)
    print('END Duramat loaded model')
    plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Duramat, class_accuracies, correct=True, data_name='Duramat', 
                                 outfolder=current_dir+output_folder)
    # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Duramat, class_accuracies, correct=False, data_name='Infinity', 
    #                              outfolder=current_dir+output_folder)


    ########## INFINITY ##########
    path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
    data_loader =  Load_Data(path)
    data = data_loader.get_data()
    val_dataset = PVDataset(data, channels=[0, 1, 2], transform=transform, scale=1)
    predictions = trainer.predict(val_dataset) 
    accuracy_Infinity = predictions.metrics['test_accuracy']
    predlabels = predictions.predictions.argmax(axis=-1)
    # Extract the true labels from val_dataset
    true_labels = np.array([label for _, label in val_dataset.df])
    # Calculate accuracy for each class
    unique_labels = np.unique(true_labels)
    class_accuracies = {}
    for label in unique_labels:
        class_accuracies[label] = calculate_class_accuracy(true_labels, predlabels, class_label=label)

    print("Class accuracies:", class_accuracies)
    plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Infinity, class_accuracies, correct=True, data_name='Infinity', 
                                 outfolder=current_dir+output_folder)
    # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Infinity, class_accuracies, correct=False, data_name='Infinity', 
    #                            outfolder=current_dir+output_folder)

    print(predictions.metrics)
    
    print('END Infinity')

    ploting_training_results(None, current_dir+output_folder, last_checkpoint=last_checkpoint, accuracies=[accuracy_Duramat,accuracy_Infinity])
