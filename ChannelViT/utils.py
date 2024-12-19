from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(predicted, orgin_labels):
    labels = orgin_labels
    preds = predicted
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1
    }