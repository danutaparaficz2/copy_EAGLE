from sklearn.metrics import f1_score, accuracy_score, log_loss
import torch
from transformers import  EvalPrediction
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

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

def plot_samples(ds_val, predlabels, correct=True, data_name='Unknown'):
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
            ax[i,j].imshow(np.transpose(s['images'], (1,2,0)))
            ax[i,j].set_title(f"G: {s['labels']}\nP: {int(predlabels[int(idx[0])])}")
            ax[i,j].axis('off')
    if correct:
        flag='correct'
    else:
        flag='wrong'
    plt.savefig('./Data/samples_'+data_name+'_'+flag+'.png')


def ploting_training_results(trainer, folder):

    # Plot the loss function for training and evaluation data

    # Extract training and validation losses
    # Extract the log history
    log_history = trainer.model.state_dict()['log_history']

    # Convert log history to DataFrame
    log_df = pd.DataFrame(log_history)

    # Extract training and validation losses
    train_losses = log_df[log_df['loss'].notna()]['loss'].values
    val_losses = log_df[log_df['eval_loss'].notna()]['eval_loss'].values        

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(folder+'./Data/loss_plot.png')
    plt.close()