import torch
import os
import numpy as np
import re
import argparse
PYTORCH_ENABLE_MPS_FALLBACK=1
#### Local imports
from load_data import AlternatingBatchSampler, PVDataset, find_outliers, load_all_data_together, Load_Data_Handler_notlabeled, just_transform, ConcatDataset
from utils import  (convert_list_of_arrays_to_labels, calculate_class_accuracy_one_hot, class_label_save, label_names, 
                    augment_underrepresented_classes, logits_to_classes, threshold_and_max)
from plots import save_images_by_label, confusion_matrix_per_class, plot_multilabel_confusion_matrix, plot_samples_from_all_labels_with_acc
from training_multi import retrain_resume_or_load_pretrained, find_wrongly_labeled_images
from sklearn.model_selection import train_test_split
import json
import pickle


def parse_args():

    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training and evaluation.')
    parser.add_argument('--retrain', type=str, default='retrain', help='retrain, resume or nothing to predict only.')
    parser.add_argument('--use_only_EL', action='store_true', default=False, help='Use only El images')
    parser.add_argument('--all_colors', action='store_true', default=True, help='Use only RGB images')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')     
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training.')     
    # parser.add_argument('--one_type_of_input_only', action='store_true', default=False, help='Flag to choose if input will be with mixed number of classes or not.')

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
    if args.all_colors:
        name_flag = 'rgb'
    else:
        name_flag = 'gray'
    if args.use_only_EL:
        channels=[0]
    else:
        if args.all_colors:
            channels=[0,1,2,3,4,5,6]
        else:
            channels=[0,1,2]
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

    ####################################     LOAD DATA          ###########################################
    data_Website, data_Website_Ralf, data_Duramat, data_Infinity = load_all_data_together(current_dir, images_folder, name_flag=name_flag, args=args)
    ####################################     DISPLAY IMAGES          ###########################################

    # integer_labels = [torch.argmax(label).item() for _, label in data_Infinity]
    # save_images_by_label(data_Infinity, integer_labels, current_dir+images_folder+'/Infinity_test_new/')
    # integer_labels = [torch.argmax(label).item() for _, label in data_Website]
    # save_images_by_label(data_Website, integer_labels, current_dir+images_folder+'/Webpage_images_Ebrar_newtt'+name_flag, flag='Website', name_flag=name_flag)

    ####################################     REMOVE WRONGLY LABELED  DATA          ###########################################

    removed_labels_path = os.path.join(current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/', 'removed_labels_inf.json')
    with open(removed_labels_path, 'r') as f:
        removed_inf = json.load(f)
    removed_labels_path_new = os.path.join(current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/', 'removed_labels_inf_new.json')
    with open(removed_labels_path_new, 'r') as f:
        removed_inf_new = json.load(f)
    removed_labels_path = os.path.join(current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/', 'removed_labels_dur.json')
    with open(removed_labels_path, 'r') as f:
        removed_dur = json.load(f)
    removed_labels_path = os.path.join(current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/', 'removed_website.json')
    with open(removed_labels_path, 'r') as f:
        removed_website = json.load(f)
    removed_labels_path = os.path.join(current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/', 'removed_website_ralf.json')
    with open(removed_labels_path, 'r') as f:
        removed_website_ralf = json.load(f)
    # Filter out data_Duramat rows whose index is in removed_dur
    data_Duramat_filtered = [item for idx, item in enumerate(data_Duramat) if idx not in removed_dur]
    # Filter out data_Infinity rows whose index is in removed_inf
    data_Infinity_filtered1 = [item for idx, item in enumerate(data_Infinity) if idx not in removed_inf]
    data_Infinity_filtered = [item for idx, item in enumerate(data_Infinity_filtered1) if idx not in  removed_inf_new]


    data_Duramat_removed = [item for idx, item in enumerate(data_Duramat) if idx  in removed_dur]
    integer_labels = [torch.argmax(label).item() for _, label in data_Duramat_removed]
    save_images_by_label(data_Duramat_removed, integer_labels, current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'/data_Duramat_removed/',  name_flag=name_flag, original_indices=removed_dur)
    data_Infinity_removed = [item for idx, item in enumerate(data_Infinity) if idx  in removed_inf]
    integer_labels = [torch.argmax(label).item() for _, label in data_Infinity_removed]
    save_images_by_label(data_Infinity_removed, integer_labels, current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'/data_Infinity_removed/',  name_flag=name_flag, original_indices=removed_inf)



    # Filter out data_Website rows whose index is in removed_website
    data_Website_filtered = [item for idx, item in enumerate(data_Website) if idx not in removed_website]
    # Filter out data_Website_Ralf rows whose index is in removed_website_ralf
    data_Website_Ralf_filtered = [item for idx, item in enumerate(data_Website_Ralf) if idx not in removed_website_ralf]

    data_Website_removed = [item for idx, item in enumerate(data_Website) if idx  in removed_website]
    data_Website_Ralf_removed = [item for idx, item in enumerate(data_Website_Ralf) if idx  in removed_website_ralf]
    data_Web_removed = data_Website_removed+data_Website_Ralf_removed


    integer_labels = [torch.argmax(label).item() for _, label in data_Website_removed]
    save_images_by_label(data_Website_removed, integer_labels, current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'/data_Website_removed/',  name_flag=name_flag, original_indices=removed_website)
    integer_labels = [torch.argmax(label).item() for _, label in data_Website_Ralf_removed]
    save_images_by_label(data_Website_Ralf_removed, integer_labels, current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'/data_Website_Ralf_removed/',  name_flag=name_flag, original_indices=removed_website_ralf)

####################################     REMOVE WRONGLY LABELED  DATA   MODEL BASED       ###########################################
    removed_labels_path = os.path.join(current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'wrongly_labeled_Duramat', 'wrong_indices.json')
    with open(removed_labels_path, 'r') as f:
        removed_Duramat = json.load(f)
    # Filter out data_Duramat rows whose index is in removed_dur
    data_Duramat_filtered_more = [item for idx, item in enumerate(data_Duramat_filtered) if idx not in removed_Duramat]

    removed_labels_path = os.path.join(current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'wrongly_labeled_Infinity', 'wrong_indices.json')
    with open(removed_labels_path, 'r') as f:
        removed_Infinity = json.load(f)
    # Filter out data_Infinity rows whose index is in removed_Infinity
    data_Infinity_filtered_more = [item for idx, item in enumerate(data_Infinity_filtered) if idx not in removed_Infinity]

    data_Duramat_removed = [item for idx, item in enumerate(data_Duramat_filtered) if idx  in removed_Duramat]
    integer_labels = [torch.argmax(label).item() for _, label in data_Duramat_removed]
    save_images_by_label(data_Duramat_removed, integer_labels, current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'/data_Duramat_removed_model/',  name_flag=name_flag, original_indices=removed_Duramat)

    data_Infinity_removed = [item for idx, item in enumerate(data_Infinity_filtered) if idx  in removed_Infinity]
    integer_labels = [torch.argmax(label).item() for _, label in data_Infinity_removed]
    save_images_by_label(data_Infinity_removed, integer_labels, current_dir+images_folder+'/REMOVED_WRONGLY_LABELED_DATA/'+'/data_Infinity_removed_model/',  name_flag=name_flag, original_indices=removed_Infinity)

    ####################################     DISPLAY IMAGES          ###########################################
    # data_Web = data_Website_filtered+data_Website_Ralf_filtered
    # integer_labels = [torch.argmax(label).item() for _, label in data_Web]
    # save_images_by_label(data_Web, integer_labels, current_dir+images_folder+'/data_Web/',  name_flag=name_flag)

    ####################################     TRANSFORM DATA          ###########################################
    # Save filtered datasets to file for reproducibility/debugging

    filtered_data = {
        "data_Duramat_filtered_more": data_Duramat_filtered_more,
        "data_Infinity_filtered_more": data_Infinity_filtered_more,
        "data_Website_filtered": data_Website_filtered,
        "data_Website_Ralf_filtered": data_Website_Ralf_filtered,
    }

    filtered_data_path = os.path.join(current_dir + images_folder, "filtered_data.pkl")

    with open(filtered_data_path, "wb") as f:
        pickle.dump(filtered_data, f)
    print("Filtered data saved to", filtered_data_path)
    
    tensor_label_list_Duramat = just_transform(data_Duramat_filtered_more, channels=[0])
    tensor_label_list_Infinity = just_transform(data_Infinity_filtered_more, channels=[0], name='infinity')
    tensor_label_list_Website = just_transform(data_Website_filtered+data_Website_Ralf_filtered, channels=channels)


    ####################################     MAKE TRAIN TEST DATASET       ###########################################

    # 1. Split each dataset separately
    train_raw_duramat, temp_duramat = train_test_split(tensor_label_list_Duramat, test_size=0.2, random_state=42)
    train_raw_Infinity, temp_Infinity = train_test_split(tensor_label_list_Infinity, test_size=0.2, random_state=42)
    val_duramat, test_duramat = train_test_split(temp_duramat, test_size=0.5, random_state=42)
    val_Infinity, test_Infinity = train_test_split(temp_Infinity, test_size=0.5, random_state=42)
    train_raw_website, temp_website = train_test_split(tensor_label_list_Website, test_size=0.2, random_state=42)
    val_website, test_website = train_test_split(temp_website, test_size=0.5, random_state=42)

    train_duramat = augment_underrepresented_classes(train_raw_duramat)
    train_Infinity = augment_underrepresented_classes(train_raw_Infinity)
    dataset_train_duramat = PVDataset(train_duramat+train_Infinity, channels=[[0]]*(len(train_duramat)+len(train_Infinity)), scale=1, return_labels=True)
    dataset_val_duramat   = PVDataset(temp_duramat+temp_Infinity,   channels=[[0]]*(len(temp_duramat)+len(temp_Infinity)),   scale=1, return_labels=True)

    train_website = augment_underrepresented_classes(train_raw_website)
    dataset_train_website = PVDataset(train_website, channels=[channels]*len(train_website), scale=1, return_labels=True)
    dataset_val_website   = PVDataset(temp_website,   channels=[channels]*len(temp_website),   scale=1, return_labels=True)
    

    batch_size = args.batch_size

    ########################################### TWO STAGE TRAINING ######################################################
    concat_train = [ConcatDataset([dataset_train_duramat]), ConcatDataset([dataset_train_website])]
    concat_val   = [ConcatDataset([dataset_val_duramat]), ConcatDataset([dataset_val_website])]

    sampler_train = [AlternatingBatchSampler(len(dataset_train_duramat), 0, batch_size), AlternatingBatchSampler(len(dataset_train_website), 0, batch_size)]
    sampler_val   = [AlternatingBatchSampler(len(dataset_val_duramat), 0, batch_size),   AlternatingBatchSampler(len(dataset_val_website), 0, batch_size)]

    if isinstance(concat_train, list):
        print("concat_train is a list")

    ########################################### ONE STAGE TRAINING ######################################################
    # # 3. Concatenate for train and validation
    # concat_train = ConcatDataset([dataset_train_duramat, dataset_train_website])
    # concat_val   = ConcatDataset([dataset_val_duramat, dataset_val_website])
    # concat_test  = ConcatDataset([dataset_test_duramat,  dataset_test_website])

    # sampler_train = AlternatingBatchSampler(len(dataset_train_duramat), len(dataset_train_website), batch_size)
    # sampler_val   = AlternatingBatchSampler(len(dataset_val_duramat),   len(dataset_val_website), batch_size)
    # sampler_test  = AlternatingBatchSampler(len(dataset_test_duramat),  len(dataset_test_website),  batch_size)

    ####################################################  TRAINING MODE   #########################################################

    trainer = retrain_resume_or_load_pretrained(args, current_dir, input_model_folder, device, output_model_folder, sampler_train=sampler_train, 
                                      sampler_val=sampler_val, concat_train=concat_train, concat_val=concat_val, channels=channels, name_flag=name_flag+'Duramat')

    dataset_7channel_Website = PVDataset(tensor_label_list_Website, channels=[[0,1,2,3,4,5,6]]*(len(tensor_label_list_Website)), scale=1, return_labels=True)
    # dataset_1channel_Infinity = PVDataset(tensor_label_list_Infinity, channels=[[0]]*(len(tensor_label_list_Infinity)), scale=1, return_labels=True)

    find_wrongly_labeled_images(
        trainer, 
        dataset_7channel_Website, 
        threshold=0.8,  # Adjust threshold as needed
        save_dir=current_dir + images_folder + "wrongly_labeled_Website/"
    )

    ####################################################  OUTLIER DETECTION MODE   #########################################################
    
    # find_outliers(tensor_label_list_Duramat, device, current_dir, trainer, threshold=5.0)
    ####################################################  PREDICION MODE   #########################################################

    predictions = trainer.predict(dataset_val_website) 
    pred_labels = logits_to_classes(predictions, initial_threshold=0.5)
    predlabels = convert_list_of_arrays_to_labels(pred_labels)

    true_labels_Web = np.array([item['labels'] for item in dataset_val_website])
    class_accuracies = {}
    print('################# ACCURACIES VALIDATION #################')
    for label in range(7):
        label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels_Web, pred_labels, class_label=label)
        print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")

    plot_samples_from_all_labels_with_acc(dataset_val_website, predlabels, class_accuracies, data_name='Website', 
                                          outfolder=current_dir+images_folder)
   #############################################  PREDICTIONS test #########################################################
    exit()
    predictions_Web = trainer.predict(dataset_test_website) 
    pred_labels = logits_to_classes(predictions_Web, initial_threshold=0.5)
    true_labels_Web = np.array([item['labels'] for item in dataset_test_website])
    class_accuracies = {}
    print('################# ACCURACIES TEST #################')
    for label in range(7):
        label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels_Web, pred_labels, class_label=label)
        print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")

    predictions_dur = trainer.predict(dataset_test_duramat) 
    pred_labels = logits_to_classes(predictions_dur, initial_threshold=0.5)
    true_labels_dur = np.array([item['labels'] for item in dataset_test_duramat])
    class_accuracies = {}
    print('################# ACCURACIES TEST #################')
    for label in range(7):
        label_name, class_accuracies[label], length = calculate_class_accuracy_one_hot(true_labels_dur, pred_labels, class_label=label)
        print(f"Class '{label_name}': accuracy={class_accuracies[label]:.2f}, count={length}")
    exit()
    # #################################################  PREDICTIONS #########################################################
    # #################################################  PREDICTIONS #########################################################
#     data_loader_3 = Load_Data_Handler(path_Website, args, classified_by=["Ebrar","Ralf"], this_folders_only=['23-P09-D'])
#     data_Website_D = data_loader_3.get_data()
#    # save_images_by_label(data_Website, data_loader_3.labels_as_integers, current_dir+images_folder+'/Webpage_images_Ebrar_test', flag='Website')

#     data_Website_D = [(item[0], item[1][0:args.num_classes]) for item in data_Website_D]    # Remove data with labels above 3, from DARK and above

#     label_counts_Website = count_data_per_class_in_labels(data_loader_3.labels_as_integers)
#     dataset_Website_D = just_transform(data_Website_D, channels=channels)
#     predictions_Web = trainer.predict(dataset_Website_D) 

#     pred_labels = logits_to_classes(predictions_Web, initial_threshold=0.5)
#     pred_labels = (pred_labels > 0.5).astype(int)
#     true_labels_Web = np.array([item['labels'] for item in dataset_Website_D])
#     confusion_matrix_per_class(pred_labels, true_labels_Web, plot=False, normalize=False)
#     class_accuracies = {}
#     for label in range(list(label_counts_Website.keys())[-1]+1):
#         class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_Web, pred_labels, class_label=label)
#         print("Class accuracies:", class_accuracies)
#     for arg, value in vars(args).items():
#         print(f"{arg}: {value}")
    # ########### DURAMAT PREDICTION !!!! ##########

    predictions_val = trainer.predict(dataset_val_duramat) 

    pred_labels = logits_to_classes(predictions_val, initial_threshold=0.5)
    # pred_labels = (pred_labels > 0.5).astype(int)
    true_labels_val = np.array([item['labels'] for item in dataset_val_duramat])
    # labels = [item[1] for item in predictions_val.label_ids]  # Extract the labels (tensors)
    # integer_labels = [np.where(label == 1)[0][0].item() for label in labels]
    # plot_multilabel_confusion_matrix(true_labels_val, pred_labels, class_names=['good','crack','cross','dark','corrosion','discoloration', 'delamination'], 
    #                                  output_path='./Data/results/confusion_matrix1.png')

    # confusion_matrix_per_class(pred_labels, true_labels_val, plot=False, normalize=False)
    class_accuracies = {}
    for label in range(7):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_val, pred_labels, class_label=label)
        print("Class accuracies Duramat:", class_accuracies)

    predictions_val = trainer.predict(dataset_test_duramat) 

    pred_labels = logits_to_classes(predictions_val, initial_threshold=0.5)
    # pred_labels = (pred_labels > 0.5).astype(int)
    true_labels_val = np.array([item['labels'] for item in dataset_test_duramat])
    # labels = [item[1] for item in predictions_val.label_ids]  # Extract the labels (tensors)
    # integer_labels = [np.where(label == 1)[0][0].item() for label in labels]
    plot_multilabel_confusion_matrix(true_labels_val, pred_labels, class_names=['good','crack','cross','dark','corrosion','discoloration', 'delamination'], 
                                     output_path='./Data/results/confusion_matrix_test_Duramat.png')

    # confusion_matrix_per_class(pred_labels, true_labels_val, plot=False, normalize=False)
    class_accuracies = {}
    for label in range(7):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_val, pred_labels, class_label=label)
        print("Class accuracies Duramat:", class_accuracies)

    # # ########### INFINITY PREDICTION !!!! ##########


    # predictions_Infinity = trainer.predict(dataset_Infinity) 
    # # accuracy_Infinity = predictions.metrics['test_accuracy']

    # pred_labels = logits_to_classes(predictions_Infinity, initial_threshold=0.5)
    # # pred_labels = (pred_labels > 0.5).astype(int)
    # true_labels_val = np.array([item['labels'] for item in dataset_Infinity])
    # plot_multilabel_confusion_matrix(true_labels_val, pred_labels, class_names=['good','crack','cross','dark','corrosion','discoloration', 'delamination'], 
    #                                  output_path='./Data/results/confusion_matrix_val_Infinity.png')
    # class_accuracies = {}
    # for label in range(7):
    #     class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_val, pred_labels, class_label=label)
    #     print("Class accuracies Infinity:", class_accuracies)
    exit()
    # ########### Web VALIDATION PREDICTION !!!! ##########
    predictions_Web = trainer.predict(val_data_web) 

    pred_labels = logits_to_classes(predictions_Web, initial_threshold=0.5)
    pred_labels = (pred_labels > 0.5).astype(int)
    true_labels_Web = np.array([item['labels'] for item in val_data_web])
    confusion_matrix_per_class(pred_labels, true_labels_Web, plot=False, normalize=False)
    class_accuracies = {}
    for label in range(list(label_counts.keys())[-1]+1):
        class_accuracies[label] = calculate_class_accuracy_one_hot(true_labels_Web, pred_labels, class_label=label)
        print("Class accuracies:", class_accuracies)


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