import torch
import os
import numpy as np
import re
import argparse
PYTORCH_ENABLE_MPS_FALLBACK=1
#### Local imports
from load_data import Load_Data, PVDataset, Load_Data_Handler
from training_multi import  init_trainer, load_model, load_post_trained_model, data_split_and_transform
from utils import  (plot_samples, ploting_training_results, count_data_per_class, plot_samples_from_all_labels, convert_list_of_arrays_to_labels,
calculate_class_accuracy_one_hot, find_last_checkpoint, calculate_class_accuracy, convert_labels_to_one_hot, 
count_data_per_class_in_labels, combine_datasets_in_batches, logits_to_classes)
from image_alignment import plot_aligned_images
from torch.utils.data import ConcatDataset, DataLoader
from training_var import CustomTrainer, train_save_model, data_just_transform
from sklearn.model_selection import train_test_split
import pickle


def parse_args():

    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training and evaluation.')
    parser.add_argument('--in_chans', type=int, default=3, help='Number of input channels.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training.')     
    parser.add_argument('--init_weights_name', type=str, default='imagenet_channelvit_small_p16_with_hcs_supervised', help='Name of the initial weights file.')
   # imagenet_channelvit_small_p16_with_hcs_supervised, so2sat_channelvit_small_p8_with_hcs_hard_split_supervised
   # cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised, camelyon_channelvit_small_p8_with_hcs_supervised
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
    output_folder = '/Data/model_with_'+args.init_weights_name+'/epochs_'+str(args.num_train_epochs)+'/'

    # Find the last checkpoint

    ######################################### Load the data ##################################################
    ########### DURAMAT ##########
    path_Duramat = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
    # Loop over a directory to read pickle files

    directory_path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/"

    # for filename in os.listdir(directory_path):
    #     if filename.endswith(".pkl"):
    #         file_path = os.path.join(directory_path, filename)
    #         with open(file_path, 'rb') as file:
    #             data = pickle.load(file)
    #             print(data.labels)
    path_Duramat = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
    data_loader =  Load_Data(path_Duramat)
    data_Duramat = data_loader.get_data()
    label_counts_duramat = count_data_per_class(data_Duramat)
    data_Duramat = convert_labels_to_one_hot(data_Duramat, len(label_counts_duramat))
    dataset_duramat = data_just_transform(data_Duramat, channels=[0])
    data_loader.get_label_statistics()
    plot_samples_from_all_labels(dataset_duramat,None, list(label_counts_duramat.keys()), data_name='dur', outfolder=current_dir+output_folder)

    ########### INFINITY ##########
    path_Infinity = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
    data_loader =  Load_Data(path_Infinity)
    data_Infinity = data_loader.get_data()
    data_loader.get_label_statistics()
    data_Infinity = [item for item in data_Infinity if item[1] <= 6]    # Remove data with labels above 3
    label_counts_infinity = count_data_per_class(data_Infinity)
    data_Infinity = convert_labels_to_one_hot(data_Infinity, len(label_counts_infinity))
    dataset_Infinity = data_just_transform(data_Infinity, channels=[0])
    data_loader.get_label_statistics()
    plot_samples_from_all_labels(dataset_Infinity,None, list(label_counts_infinity.keys()), data_name='inf', outfolder=current_dir+output_folder)

    ########### WEBSITE ##########

    path_Website = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    data_loader_2 = Load_Data_Handler(path_Website)
    data_Website = data_loader_2.get_data()
    data_Website = [(item[0], item[1][0:4]) for item in data_Website]    # Remove data with labels above 3
    label_counts_Website = count_data_per_class_in_labels(data_loader_2.labels_as_integers)
    dataset_Website = data_just_transform(data_Website, channels=[0, 1, 2])
    plot_samples_from_all_labels(dataset_Website,None, list(label_counts_duramat.keys()), data_name='web', outfolder=current_dir+output_folder)

    ########### COMBINE DATASETS ##########
    label_counts = {key: label_counts_duramat.get(key, 0) + label_counts_infinity.get(key, 0) + 
                    label_counts_Website.get(key, 0) for key in set(label_counts_duramat) | set(label_counts_infinity)| set(label_counts_Website)}

    datas = (dataset_Infinity+dataset_duramat)
    train_data, val_data = train_test_split(datas, test_size=0.3, random_state=42)
    train_data_web, val_data_web = train_test_split(dataset_Website, test_size=0.3, random_state=42)
    # train_data = combine_datasets_in_batches(train_data, train_data_web, batch_size=args.num_train_epochs)
    # val_data = combine_datasets_in_batches(val_data, val_data_web, batch_size=args.num_train_epochs)

    #################################################### ONLY TRAINING MODE  #########################################################
    # Model with originally pretrained weights
    model = load_model(args, current_dir+'/Data/', device, args.init_weights_name)
    trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_folder+'/all/')
    trainer.train()
    trainer = train_save_model(trainer, current_dir+output_folder+'/all/')

    ###################################### VALIDATION ########################################################
    # ########## Duramat + Infinity ##########
    model = load_post_trained_model(args, current_dir+output_folder+'/all/', device, 'trained_state_dict')
    trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_folder+'/all/')
    predictions = trainer.predict(val_data)
    true_labels = np.array([item['labels'] for item in val_data])
    pred_labels = logits_to_classes(predictions.cpu().numpy())
    predlabels = convert_list_of_arrays_to_labels(pred_labels)

    class_accuracies= {}
    for label in range(len(label_counts_duramat)):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions.cpu().numpy(), class_label=label)
    print("Class accuracies:", class_accuracies)
    print('END Duramat + Inifinity prediction')
    
    plot_samples_from_all_labels(val_data, predlabels, list(label_counts_duramat.keys()), data_name='Duramat+Infinity', outfolder=current_dir+output_folder)

    # ########## Webpage ##########
    predictions = trainer.predict(val_data_web)
    true_labels = np.array([item['labels'] for item in val_data_web])
    # pred_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    pred_labels = logits_to_classes(predictions.cpu().numpy())
    predlabels = convert_list_of_arrays_to_labels(pred_labels)

    class_accuracies= {}
    for label in range(len(label_counts_duramat)):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions.cpu().numpy(), class_label=label)
    print("Class accuracies:", class_accuracies)
    print('END Webpage prediction')
    plot_samples_from_all_labels(val_data_web, predlabels, list(label_counts_duramat.keys()), data_name='Webpage', outfolder=current_dir+output_folder)

    exit()
    # trainer = init_trainer(args, model, val_data_web, current_dir+output_folder+'/all/')
    # trainer = train_save_model(trainer, train_data_web, val_data_web, current_dir+output_folder+'/all/')
    # ploting_training_results(trainer, current_dir+output_folder+'/all/')
    
    # ########################################### ONLY PREDICT ###########################################
    # # ########## Duramat ##########
    # # # Load the trained model back into the trainer
    # model = load_post_trained_model(args, current_dir+output_folder+'/all/', device, 'trained_state_dict')

    # #This method is used to set the model to evaluation mode. It is important to call this method before running inference, 
    # # because the model needs to know that it is in evaluation mode so that it can turn off features like dropout and batch normalization.
    # model.eval()
    # trainer = init_trainer(args, model, val_data, current_dir+output_folder)
    # # Use the trainer to make predictions
    # predictions = trainer.predict(val_data)
    # predlabels = convert_list_of_arrays_to_labels(predictions.label_ids)
    # # Extract the true labels from val_dataset
    # true_labels = np.array([item['labels'] for item in val_data])
    # print(predictions.metrics)
    # accuracy_Duramat = predictions.metrics['test_accuracy']
    # # Calculate accuracy for each class
    # class_accuracies = {}
    # for label in range(len(label_counts)):
    #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions[0], class_label=label)
   
    # print("Class accuracies:", class_accuracies)
    # print('END Duramat loaded model')
    # plot_samples_from_all_labels(val_data, predlabels, list(label_counts.keys()), data_name='Duramat', outfolder=current_dir+output_folder)
    # # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Duramat, class_accuracies, correct=False, data_name='Infinity', 
    # #                              outfolder=current_dir+output_folder)
    # exit()
    # # ########## INFINITY ##########
    # path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
    # data_loader =  Load_Data(path)
    # data = data_loader.get_data()
    # data_loader.get_label_statistics()

    # # Remove data with labels above 3
    # data = [item for item in data if item[1] <= 3]
    # data_loader.get_label_statistics()
    # label_counts = count_data_per_class(data)
    # data = convert_labels_to_one_hot(data)
    # train_dataset, val_dataset, transform = data_split_and_transform(data)
    # labels_data = [label for _, label in data]


    # ###########TRAIN INFINITY################
    # # model = load_post_trained_model(args, current_dir+output_folder, device, 'trained_state_dict')
    # # trainer = init_trainer(args, model, val_dataset, current_dir+output_folder)
    # # trainer = train_save_model(trainer, train_dataset, val_dataset, current_dir+output_folder+'retrained_on_infinity/')
    # # ploting_training_results(trainer, current_dir+output_folder)
    # #########################################
    # # model = load_model(args, current_dir+'/Data/', device, args.init_weights_name)
    # model = load_post_trained_model(args, current_dir+output_folder+'/all/', device, 'trained_state_dict')
    # model.eval()
    # # val_dataset = PVDataset(data, channels=[0, 1, 2], transform=transform, scale=1)
    # trainer = init_trainer(args, model, val_dataset, current_dir+output_folder)

    # predictions = trainer.predict(val_dataset) 
    # accuracy_Infinity = predictions.metrics['test_accuracy']
    # predlabels = predictions.predictions.argmax(axis=-1)
    # # Extract the true labels from val_dataset
    # true_labels = np.array([label for _, label in val_dataset.df])
    # # Calculate accuracy for each class
    # class_accuracies = {}
    # for label in range(len(label_counts)):
    #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions[0], class_label=label)

    # print("Class accuracies:", class_accuracies)
    # plot_samples_from_all_labels(val_dataset, predlabels, list(label_counts.keys()), data_name='Infinity',  outfolder=current_dir+output_folder)
    # # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Infinity, class_accuracies, correct=False, data_name='Infinity', 
    # #                            outfolder=current_dir+output_folder)

    # print(predictions.metrics)
    
    # print('END Infinity')

    # ########## WEBPAGE ##########



    # accuracy_Infinity = predictions.metrics['test_accuracy']
    # predlabels = predictions.predictions.argmax(axis=-1)
    # # Extract the true labels from val_dataset
    # true_labels = np.array([label for _, label in val_dataset.df])
    # # Calculate accuracy for each class
    # class_accuracies = {}
    # for label in range(len(label_counts)):
    #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions[0], class_label=label)

    # print("Class accuracies:", class_accuracies)
    # plot_samples_from_all_labels(val_dataset, predlabels,  list(label_counts.keys()), data_name='WEBPAGE',  outfolder=current_dir+output_folder+'/all/')
    # # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Infinity, class_accuracies, correct=False, data_name='Infinity', 
    # #                            outfolder=current_dir+output_folder)

    # print(predictions.metrics)
    