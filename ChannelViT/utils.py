from sklearn.metrics import f1_score, accuracy_score

from transformers import TrainingArguments, Trainer, EvalPrediction


def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    loss = p.predictions[1]
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
        'loss': loss.mean()  # Assuming loss is a tensor, take the mean
    }