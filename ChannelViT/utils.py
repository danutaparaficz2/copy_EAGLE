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

def plot_samples(ds_val, predictions, predlabels, correct=True):
    fig, ax = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(20,20))
    idx = -1
    for i in range(6):
        for j in range(6):
            while True:
                if correct:
                    idx = np.random.choice(ds_val.shape[0], 1, replace=False)
                    if ds_val[int(idx[0])]["labels"] > 0 and ds_val[int(idx[0])]["labels"] == predlabels[int(idx[0])]:
                        break
                else:
                    for id in np.where(ds_val['labels'] != predlabels)[0]:
                        if id > idx:
                            idx = int(id)
                            break
                    if idx > -1:
                        break
            s = ds_val[int(idx[0])]
            ax[i,j].imshow(np.transpose(s['image'], (1,2,0)))
            ax[i,j].set_title(f"G: {s['labels']}\nP: {predlabels[int(idx[0])]}")
            ax[i,j].axis('off')
    plt.show()


def ploting_training_results():

    # Plot the loss function for training and evaluation data
    with open('/Users/eagle/Documents/eagle-classification/Data/working/checkpoint-48/trainer_state.json', 'r') as f:
        logs = json.load(f)
    # Extract training and validation losses
    # Extract the log history
    log_history = logs['log_history']

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
    plt.savefig('/Users/eagle/Documents/eagle-classification/Data/loss_plot.png')
    plt.close()