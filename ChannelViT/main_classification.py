import torch
import os
import numpy as np
import re
import argparse
PYTORCH_ENABLE_MPS_FALLBACK=1
#### Local imports
from load_data import Load_Data, PVDataset, Load_Data_Handler, Load_Data_Handler_notlabeled
from utils import  (count_data_per_class, convert_list_of_arrays_to_labels, calculate_class_accuracy_one_hot, class_label_save, 
convert_labels_to_one_hot, label_names, augment_underrepresented_classes, logits_to_classes, plot_normalized_confusion_matrix,ploting_training_results,
count_data_per_class_in_labels, threshold_and_max, save_images_by_label, count_data_per_multiclass, plot_samples_from_all_labels_with_acc)
from training_var import CustomTrainer, train_save_model, data_just_transform, load_model, load_post_trained_model
from training_multi_obsolete import CustomTrainer, train_save_model, data_just_transform, init_trainer

from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from PIL import Image
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator

def parse_args():

    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=29, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training and evaluation.')
    parser.add_argument('--retrain', action='store_true', default=True, help='Flag to retrain the model.')
    parser.add_argument('--use_only_EL', action='store_true', default=True, help='Use only El images')

    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')     
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training.')     
    parser.add_argument('--one_type_of_input_only', action='store_true', default=True, help='Flag to choose if input will be with mixed number of classes or not.')

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
    output_model_folder = '/Data/models/model_with_'+args.init_weights_name+'/epochs_'+str(args.num_train_epochs)+'/'
    input_model_folder = '/Data/models/'
    images_folder = '/Data/images/'
    if args.init_weights_name == 'imagenet_channelvit_small_p16_with_hcs_supervised':
        args.max_channels = 3
    elif args.init_weights_name == 'so2sat_channelvit_small_p8_with_hcs_hard_split_supervised':
        args.max_channels = 18 
    elif args.init_weights_name == 'cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised':
        args.max_channels = 8 
    elif args.init_weights_name == 'camelyon_channelvit_small_p8_with_hcs_supervised':
        args.max_channels = 3 
    else:
        raise ValueError(f"Unknown init_weights_name: {args.init_weights_name}. Please set max_channels accordingly.")
    ######################################### Load the data ##################################################
    ########### DURAMAT ##########
    # path_Duramat = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
    # # Loop over a directory to read pickle files

    # directory_path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/"

    # path_Duramat = "/Users/eagle/FFHS/eagle-bfe - data/Duramat_no_pool_labels.pkl"
    # data_loader =  Load_Data(path_Duramat)
    # data_Duramat = data_loader.get_data()
    # label_counts_duramat = count_data_per_class(data_Duramat)
    # count_data_per_multiclass(data_Duramat)
    # labels = [item[1] for item in data_Duramat]  # Extract the labels (tensors)
    # data_Duramat = convert_labels_to_one_hot(data_Duramat, num_classes=args.num_classes)

    # # remove DARK class because in Duramat it means something different
    # for i, item in enumerate(data_Duramat):
    #     item[1][3]=0
    #     data_Duramat[i] = (item[0], item[1]) 

    # # Extract the labels (tensors)
    # labels = [item[1] for item in data_Duramat] 
    # integer_labels = [torch.argmax(item[1]).item() for item in data_Duramat]  # Convert one-hot encoded labels to integers
   
    # # save_images_by_label(data_Duramat, integer_labels, current_dir+images_folder+'/Duramat_images/')
     
    # dataset_duramat = data_just_transform(data_Duramat, channels=[0])
    # # plot_samples_from_all_labels(dataset_duramat,None, list(label_counts_duramat.keys()), data_name='dur', outfolder=current_dir+images_folder)

    # train_data, val_data = train_test_split(dataset_duramat, test_size=0.3, random_state=42)

    # label_counts = {label: 0 for label in range(len(train_data[0]['labels']))}
    # for data in train_data:
    #     for label, value in enumerate(data['labels']):
    #         if value == 1:
    #             label_counts[label] += 1

    # print(label_counts)

    # train_data = augment_underrepresented_classes(train_data, label_counts)

    # ########### WEBSITE ##########

    path_Website = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    data_loader_2 = Load_Data_Handler(path_Website, args, classified_by=["Ebrar", 'Ralf'], folders_excluded=['23-P09-D'])
    data_Website = data_loader_2.get_data()
    label_counts_Website = count_data_per_class_in_labels(data_loader_2.labels_as_integers)

    # save_images_by_label(data_Website, data_loader_2.labels_as_integers, current_dir+images_folder+'/Webpage_images', flag='Website')

    data_Website = [(item[0], item[1][0:args.num_classes]) for item in data_Website]    # Remove data with labels above 3, from DARK and above

    # path_Website = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    # data_loader_2 = Load_Data_Handler(path_Website, classified_by="Ralf", folder_excluded='23-P09-D')
    # data_Website_Ralf = data_loader_2.get_data()
    # save_images_by_label(data_Website_Ralf, data_loader_2.labels_as_integers, current_dir+images_folder+'/Webpage_images_Ralf', flag='Website')

    # data_Website_Ralf = [(item[0], item[1][0:args.num_classes]) for item in data_Website_Ralf]    # Remove data with labels above 3, from DARK and above

    # # Combine both Ebrar and Ralf lists
    
    # data_Website = data_Website_Ebrar + data_Website_Ralf
    # save_images_by_label(data_Website, data_loader_2.labels_as_integers, current_dir+images_folder+'/Webpage_images_Ebrar_Ralf', flag='Website')    #
    if args.use_only_EL:
        channels=[0]
    else:
        channels=[0,1,2]
    dataset_Website = data_just_transform(data_Website, channels=channels)
    train_data_web, val_data_web = train_test_split(dataset_Website, test_size=0.3, random_state=42)
    train_data_web = augment_underrepresented_classes(train_data_web, label_counts_Website)


    #################################################### ONLY TRAINING MODE DURAMAT  #########################################################
    # Model with originally pretrained weights
    if args.one_type_of_input_only:
        if args.retrain:
            model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
            trainer = init_trainer(args, model, val_data_web, current_dir+output_model_folder)
            trainer = train_save_model(trainer, train_data_web, val_data_web, current_dir+output_model_folder+'website_'+str(len(channels))+'classes_var_new/')
            ploting_training_results(trainer, current_dir+output_model_folder+'website_'+str(len(channels))+'classes_var_new/')
        else:
            model = load_post_trained_model(args, current_dir+output_model_folder+'website_'+str(len(channels))+'classes_var_new/', device, 'trained_state_dict', args.num_classes)
            model.eval()
            trainer = init_trainer(args, model, val_data_web, current_dir+output_model_folder+'website_'+str(len(channels))+'classes_var_new/')

    else:
        if args.retrain:
            model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name, args.num_classes)
            trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_model_folder+'/duramat_4classes_var_new_fixed_transformed_with_dark/')
            trainer.train()
            trainer = train_save_model(trainer, current_dir+output_model_folder+'duramat_4classes_var_new_fixed_transformed_with_dark/')
            # # ploting_training_results(trainer, current_dir+output_model_folder+'duramat_3classes_var/')
        else:
            model = load_post_trained_model(args, current_dir+output_model_folder+'/duramat_4classes_var_new_fixed_transformed_with_dark/', device, 'trained_state_dict', args.num_classes)
            trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_model_folder+'/duramat_4classes_var_new_fixed_transformed_with_dark/')

    # #################################################  PREDICTIONS #########################################################
    predictions_Web = trainer.predict(val_data_web) 
    pred_labels = logits_to_classes(predictions_Web[0], initial_threshold=0.5)
    pred_labels = (pred_labels > 0.5).astype(int)
    true_labels_Web = np.array([item['labels'] for item in val_data_web])
    class_accuracies = {}
    for label in range(list(label_counts_Website.keys())[-1]+1):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_Web, pred_labels, class_label=label)
        print("Class accuracies:", class_accuracies)

    # #################################################  PREDICTIONS #########################################################
    # #################################################  PREDICTIONS #########################################################
    data_loader_3 = Load_Data_Handler(path_Website, classified_by=["Ebrar"], this_folders_only=['23-P09-D'])
    data_Website_D = data_loader_3.get_data()
   # save_images_by_label(data_Website, data_loader_3.labels_as_integers, current_dir+images_folder+'/Webpage_images_Ebrar_test', flag='Website')

    data_Website_D = [(item[0], item[1][0:args.num_classes]) for item in data_Website_D]    # Remove data with labels above 3, from DARK and above

    label_counts_Website = count_data_per_class_in_labels(data_loader_3.labels_as_integers)
    dataset_Website_D = data_just_transform(data_Website_D, channels=channels)


    predictions_Web = trainer.predict(dataset_Website_D) 
    pred_labels = logits_to_classes(predictions_Web[0], initial_threshold=0.5)
    pred_labels = (pred_labels > 0.5).astype(int)
    true_labels_Web = np.array([item['labels'] for item in dataset_Website_D])
    class_accuracies = {}
    for label in range(list(label_counts_Website.keys())[-1]+1):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_Web, pred_labels, class_label=label)
        print("Class accuracies:", class_accuracies)
    exit()
    # ########### INFINITY PREDICTION !!!! ##########

    path_Infinity = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
    data_loader =  Load_Data(path_Infinity)
    data_Infinity = data_loader.get_data()
    data_loader.get_label_statistics()
    data_Infinity = [item for item in data_Infinity if item[1] <= args.num_classes-1]    # Remove data with labels above 2 - dark class is defined differently in Inifinity than duramat

    label_counts_infinity = count_data_per_class(data_Infinity)
    count_data_per_multiclass(data_Infinity)

    data_Infinity = convert_labels_to_one_hot(data_Infinity, len(label_counts_infinity))
    dataset_Infinity = data_just_transform(data_Infinity, channels=[0])
    data_loader.get_label_statistics()
    labels = [item[1] for item in data_Infinity]  # Extract the labels (tensors)
    integer_labels = [torch.where(label == 1)[0][0].item() for label in labels]
    save_images_by_label(data_Infinity, integer_labels, current_dir+images_folder+'/Infinity_images_with_dark/')
    predictions_Infinity = trainer.predict(dataset_Infinity) 
    # accuracy_Infinity = predictions.metrics['test_accuracy']


    binary_arr = threshold_and_max(predictions_Infinity.cpu().numpy())
    predlabels_Infinity_multi = convert_list_of_arrays_to_labels(binary_arr)
    predlabels_Infinity = predictions_Infinity.argmax(axis=-1).cpu().numpy()
    #  predlabels = predictions.predictions.argmax(axis=-1)
    # Extract the true labels from val_dataset
    true_labels_Infinity = np.array([label for _, label in dataset_Infinity.df])
    # Calculate accuracy for each class
    class_accuracies = {}
    for label in range(len(label_counts_infinity)):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_Infinity, binary_arr, class_label=label)
        print("Class accuracies:", class_accuracies)
    
    # plot_normalized_confusion_matrix(integer_labels, predlabels_Infinity, class_names=['good','crack','cross'], output_path='.Data/results/confusion_matrix.png')
    # plot_samples_from_all_labels_with_acc(dataset_Infinity, predlabels_Infinity_multi, class_accuracies, data_name='Infinity', outfolder='./Data')
    ########################################### TISO ###########################################

    # path_Website = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    # data_loader_2  = Load_Data_Handler_notlabeled(path_Website,'TISO-EAGLE-23-P09_images')
    # data_TISO, im_names = data_loader_2.get_data()
    # dataset_TISO = data_just_transform(data_TISO, channels=[0, 1, 2] , return_labels=False)

    # predictions_TISO = trainer.predict(dataset_TISO)
    # # Save predictions to a parquet file
    # predlabels_TISO = predictions_TISO.cpu().numpy().argmax(axis=-1)    

    # binary_arr = threshold_and_max(predictions_TISO.cpu().numpy())
    # predlabels_TISO_multi = convert_list_of_arrays_to_labels(binary_arr)

    # save_images_by_label(data_TISO, predlabels_TISO_multi, current_dir+images_folder+'./TISO_images_new/')
    # class_label_save(predlabels_TISO, im_names, label_names() ,'predictions_tiso.parquet')


    # ########################################### C14 ###########################################
    circolo14 = 'C14-J'
    data_loader_2  = Load_Data_Handler_notlabeled(path_Website, circolo14)
    data_C14, im_names = data_loader_2.get_data()
    dataset_C14 = data_just_transform(data_C14, channels=[0, 1, 2] , return_labels=False)

    predictions_C14 = trainer.predict(dataset_C14)
    # Save predictions to a parquet file
    predlabels_C14 = predictions_C14.cpu().numpy().argmax(axis=-1)

    binary_arr = threshold_and_max(predictions_C14.cpu().numpy())
    predlabels_C14_multi = convert_list_of_arrays_to_labels(binary_arr)

    save_images_by_label(data_C14, predlabels_C14_multi, current_dir+images_folder+circolo14+'_images_new//', flag='Website')
    class_label_save(predlabels_C14, im_names, label_names(),'predictions_'+circolo14+'.parquet')

    exit()
    # # ########## Webpage ##########
    # predictions = trainer.predict(val_data)
    # true_labels = np.array([item['labels'] for item in val_data])
    # # pred_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    # pred_labels = logits_to_classes(predictions.cpu().numpy())
    # predlabels = convert_list_of_arrays_to_labels(pred_labels)

    # class_accuracies= {}
    # for label in range(len(label_counts_duramat)):
    #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions.cpu().numpy(), class_label=label)
    # print("Class accuracies:", class_accuracies)
    # print('END Webpage prediction')
    # plot_samples_from_all_labels(val_data, predlabels, list(label_counts_duramat.keys()), data_name='Duramat', outfolder=current_dir+images_folder)


    # # print("Class accuracies:", class_accuracies)
    # # plot_samples_from_all_labels(dataset_Infinity,None, list(label_counts_infinity.keys()), data_name='inf', outfolder=current_dir+images_folder)
    # # # Separate images and labels into two lists

    # # ########### WEBSITE ##########

    # path_Website = "/Users/eagle/Library/CloudStorage/OneDrive-SharedLibraries-FFHS/eagle-bfe - data/Webpage"
    # data_loader_2 = Load_Data_Handler(path_Website)
    # data_Website = data_loader_2.get_data()
    # data_Website = [(item[0], item[1][0:4]) for item in data_Website]    # Remove data with labels above 3
    #     # combine class 2 and 3 (they are the same in Duramat data (see presentation))
    # data_Website = [(item[0], item[1] if item[1] != 3 else 2) for item in data_Website]
    # # Remove data with labels above 3
    # data_Website = [item for item in data_Website if item[1] <= 3]
    # label_counts_Website = count_data_per_class_in_labels(data_loader_2.labels_as_integers)
    # dataset_Website = data_just_transform(data_Website, channels=[0, 1, 2])
    # # plot_samples_from_all_labels(dataset_Website,None, list(label_counts_duramat.keys()), data_name='web', outfolder=current_dir+images_folder)

    # # save_images_by_label(data_Website, data_loader_2.labels_as_integers, current_dir+images_folder+'/Webpage_images_new/')

    # ########### COMBINE DATASETS ##########
    # # label_counts = {key: label_counts_duramat.get(key, 0) + label_counts_infinity.get(key, 0) + 
    # #                 label_counts_Website.get(key, 0) for key in set(label_counts_duramat) | set(label_counts_infinity)| set(label_counts_Website)}

    # datas = (dataset_duramat)
    # train_data, val_data = train_test_split(datas, test_size=0.3, random_state=42)
    # # train_data_web, val_data_web = train_test_split(dataset_Website, test_size=0.3, random_state=42)
    # # train_data = combine_datasets_in_batches(train_data, train_data_web, batch_size=args.num_train_epochs)
    # # val_data = combine_datasets_in_batches(val_data, val_data_web, batch_size=args.num_train_epochs)

    # #################################################### ONLY TRAINING MODE  #########################################################
    # # Model with originally pretrained weights
    # model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name)
    # trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_model_folder+'/all/')
    # trainer.train()
    # trainer = train_save_model(trainer, current_dir+output_model_folder+'/all/')


    # # ########## Duramat + Infinity ##########
    # model = load_post_trained_model(args, current_dir+output_model_folder+'/all/', device, 'trained_state_dict')
    # trainer = CustomTrainer(model, args, train_data, train_data_web, val_data, val_data_web, device,  current_dir+output_model_folder+'/all/')

    # true_labels = np.array([item['labels'] for item in val_data])
    # pred_labels = logits_to_classes(predictions.cpu().numpy())
    # predlabels = convert_list_of_arrays_to_labels(pred_labels)

    # class_accuracies= {}
    # for label in range(len(label_counts_duramat)):
    #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions.cpu().numpy(), class_label=label)
    # print("Class accuracies:", class_accuracies)
    # print('END Duramat + Inifinity prediction')
    
    # plot_samples_from_all_labels(val_data, predlabels, list(label_counts_duramat.keys()), data_name='Duramat+Infinity', outfolder=current_dir+images_folder)

    # # ########## Webpage ##########
    # predictions = trainer.predict(val_data_web)
    # true_labels = np.array([item['labels'] for item in val_data_web])
    # # pred_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    # pred_labels = logits_to_classes(predictions.cpu().numpy())
    # predlabels = convert_list_of_arrays_to_labels(pred_labels)

    # class_accuracies= {}
    # for label in range(len(label_counts_duramat)):
    #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions.cpu().numpy(), class_label=label)
    # print("Class accuracies:", class_accuracies)
    # print('END Webpage prediction')
    # plot_samples_from_all_labels(val_data_web, predlabels, list(label_counts_duramat.keys()), data_name='Webpage', outfolder=current_dir+images_folder)


    # # trainer = init_trainer(args, model, val_data_web, current_dir+output_model_folder+'/all/')
    # # trainer = train_save_model(trainer, train_data_web, val_data_web, current_dir+output_model_folder+'/all/')
    # # ploting_training_results(trainer, current_dir+images_folder+'/all/')
    
    # # ########################################### ONLY PREDICT ###########################################
    # # # ########## Duramat ##########
    # # # # Load the trained model back into the trainer
    # # model = load_post_trained_model(args, current_dir+output_model_folder+'/all/', device, 'trained_state_dict')

    # # #This method is used to set the model to evaluation mode. It is important to call this method before running inference, 
    # # # because the model needs to know that it is in evaluation mode so that it can turn off features like dropout and batch normalization.
    # # model.eval()
    # # trainer = init_trainer(args, model, val_data, current_dir+output_model_folder)
    # # # Use the trainer to make predictions
    # # predictions = trainer.predict(val_data)
    # # predlabels = convert_list_of_arrays_to_labels(predictions.label_ids)
    # # # Extract the true labels from val_dataset
    # # true_labels = np.array([item['labels'] for item in val_data])
    # # print(predictions.metrics)
    # # accuracy_Duramat = predictions.metrics['test_accuracy']
    # # # Calculate accuracy for each class
    # # class_accuracies = {}
    # # for label in range(len(label_counts)):
    # #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions[0], class_label=label)
   
    # # print("Class accuracies:", class_accuracies)
    # # print('END Duramat loaded model')
    # # plot_samples_from_all_labels(val_data, predlabels, list(label_counts.keys()), data_name='Duramat', outfolder=current_dir+images_folder)
    # # # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Duramat, class_accuracies, correct=False, data_name='Infinity', 
    # # #                              outfolder=current_dir+images_folder)
    # # exit()
    # # # ########## INFINITY ##########
    # # path = os.path.dirname(current_dir)+"/eagle-labelling/features_pickle/Infinity_all_no_pool_labels.pkl"
    # # data_loader =  Load_Data(path)
    # # data = data_loader.get_data()
    # # data_loader.get_label_statistics()

    # # # Remove data with labels above 3
    # # data = [item for item in data if item[1] <= 3]
    # # data_loader.get_label_statistics()
    # # label_counts = count_data_per_class(data)
    # # data = convert_labels_to_one_hot(data)
    # # train_dataset, val_dataset, transform = data_split_and_transform(data)
    # # labels_data = [label for _, label in data]


    # # ###########TRAIN INFINITY################
    # # # model = load_post_trained_model(args, current_dir+output_model_folder, device, 'trained_state_dict')
    # # # trainer = init_trainer(args, model, val_dataset, current_dir+output_model_folder)
    # # # trainer = train_save_model(trainer, train_dataset, val_dataset, current_dir+output_model_folder+'retrained_on_infinity/')
    # # # ploting_training_results(trainer, current_dir+output_model_folder)
    # # #########################################
    # # # model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name)
    # # model = load_post_trained_model(args, current_dir+output_model_folder+'/all/', device, 'trained_state_dict')
    # # model.eval()
    # # # val_dataset = PVDataset(data, channels=[0, 1, 2], transform=transform, scale=1)
    # # trainer = init_trainer(args, model, val_dataset, current_dir+output_model_folder)

    # # predictions = trainer.predict(val_dataset) 
    # # accuracy_Infinity = predictions.metrics['test_accuracy']
    # # predlabels = predictions.predictions.argmax(axis=-1)
    # # # Extract the true labels from val_dataset
    # # true_labels = np.array([label for _, label in val_dataset.df])
    # # # Calculate accuracy for each class
    # # class_accuracies = {}
    # # for label in range(len(label_counts)):
    # #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions[0], class_label=label)

    # # print("Class accuracies:", class_accuracies)
    # # plot_samples_from_all_labels(val_dataset, predlabels, list(label_counts.keys()), data_name='Infinity',  outfolder=current_dir+images_folder)
    # # # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Infinity, class_accuracies, correct=False, data_name='Infinity', 
    # # #                            outfolder=current_dir+images_folder)

    # # print(predictions.metrics)
    
    # # print('END Infinity')

    # # ########## WEBPAGE ##########



    # # accuracy_Infinity = predictions.metrics['test_accuracy']
    # # predlabels = predictions.predictions.argmax(axis=-1)
    # # # Extract the true labels from val_dataset
    # # true_labels = np.array([label for _, label in val_dataset.df])
    # # # Calculate accuracy for each class
    # # class_accuracies = {}
    # # for label in range(len(label_counts)):
    # #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels, predictions[0], class_label=label)

    # # print("Class accuracies:", class_accuracies)
    # # plot_samples_from_all_labels(val_dataset, predlabels,  list(label_counts.keys()), data_name='WEBPAGE',  outfolder=current_dir+images_folder+'/all/')
    # # # plot_samples_from_all_labels(val_dataset, predlabels, accuracy_Infinity, class_accuracies, correct=False, data_name='Infinity', 
    # # #                            outfolder=current_dir+images_folder)

    # # print(predictions.metrics)
    


    #     ####################################################CROSS VALIDATION#########################################################
    # # model = load_model(args, current_dir+input_model_folder, device, args.init_weights_name)
    # # trainer = CustomTrainer(model, args, train_data, train_data, None, None, device,  current_dir+output_model_folder+'/all/')
    # # model.eval()

    # # class TorchModelWrapper(BaseEstimator):
    # #     def __init__(self, trainer):
    # #         self.trainer = trainer

    # #     def fit(self, X, y=None):
    # #         return self

    # #     def predict(self, X):
    # #         predictions = self.trainer.predict(X)
    # #         return logits_to_classes(predictions.cpu().numpy())

    # # # Wrap the trainer in a scikit-learn compatible estimator
    # # model_wrapper = TorchModelWrapper(trainer)

    # # # Perform cross-validation predictions
    # # predictions = cross_val_predict(model_wrapper, datas, y=None, cv=5)
    # # from cleanlab.filter import find_label_issues

    # # # Returns indices of likely label errors
    # # label_issues = find_label_issues(
    # #     labels=integer_labels,
    # #     pred_probs=predictions,
    # #     return_indices_ranked_by="self_confidence"  # low model confidence
    # # )

    # # print(f"Found {len(label_issues)} potential label issues:\n")
    # # label_folder = 'ISSUES'
    # # os.makedirs(label_folder, exist_ok=True)  # Ensure the directory exists
    # # for i in label_issues:
        
    # #     image_array = datas[i]['images'].cpu().numpy().squeeze()
    # #     # Ensure the array is in the correct format (uint8)
    # #     if image_array.min() < 0 or image_array.max() <= 1:  # If normalized to [-1, 1] or [0, 1]
    # #         image_array = ((image_array + 1) * 127.5).astype(np.uint8)  # Scale to [0, 255]
    # #     else:
    # #         image_array = image_array.astype(np.uint8)   
    # #     # Save the image
    # #     image = Image.fromarray(image_array)     
    # #     image.save(os.path.join(label_folder, f'image_{i}_{label_names()[integer_labels[i]]}.png'))
    # # ###################################### VALIDATION ########################################################