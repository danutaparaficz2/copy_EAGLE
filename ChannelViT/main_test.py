import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import pandas as pd
PYTORCH_ENABLE_MPS_FALLBACK=1
from hubconf import camelyon_channelvit_small_p8_with_hcs_supervised, so2sat_channelvit_small_p8_with_hcs_random_split_supervised
#### Local imports
from load_data import Load_Data, PVDataset
from utils import compute_metrics
from transformers import TrainingArguments, Trainer



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs['labels']
        outputs = model(inputs['images'], extra_tokens=inputs)
        loss = self.label_smoother(outputs, labels) if self.label_smoother is not None else self.compute_loss_function(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_loss_function(self, outputs, labels):
        return torch.nn.functional.cross_entropy(outputs, labels)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        labels = inputs['labels']
        with torch.no_grad():
            outputs = model(inputs['images'], extra_tokens=inputs)
        if prediction_loss_only:
            loss = self.compute_loss_function(outputs, labels)
            return (loss, None, None)
        return (None, outputs, labels)
    
# we need to collate the data to be able to use multiple inputs images and labels and channels
def custom_collate_fn(examples):
    images = torch.stack([example['images'] for example in examples])
    rest = {k: default_collate([example[k] for example in examples]) for k in examples[0] if k != 'images'}
    return {'images': images, **rest}
    

if __name__ == '__main__':
    # Usage
    path = "/Users/danuta.paraficz/PycharmProjects/eagle-classification/Data/Duramat_no_pool_labels.pkl"
    data_loader =  Load_Data(path)
    data = data_loader.get_data()
# this is an alternative way of loading the pretrained model
    # model = torch.hub.load('insitro/ChannelViT', 'imagenet_channelvit_small_p16_with_hcs_supervised', pretrained=True, map_location=torch.device('cpu'))

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.9, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_dataset = PVDataset(train_data, channels=[0, 1, 2], transform=transform, scale=1)
    val_dataset = PVDataset(val_data, channels=[0, 1, 2], transform=transform, scale=1)


    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Load the model
    model = so2sat_channelvit_small_p8_with_hcs_random_split_supervised(pretrained=False)

    # Load the pretrained weights and map them to the appropriate device
    state_dict = torch.load('Data/so2sat_channelvit_small_p8_with_hcs_hard_split_supervised.pth', map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model.to(device)
    batch_size = 10
    logging_steps = len(train_data) // batch_size
    training_args = TrainingArguments(output_dir='./working/',
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    evaluation_strategy='epoch',
                                    save_strategy='epoch',
                                    num_train_epochs=2,
                                    fp16=True if torch.cuda.is_available() else False,
                                    logging_steps=logging_steps,
                                    learning_rate=1e-5,
                                    save_total_limit=2,
                                    remove_unused_columns=False,
                                    push_to_hub=False,
                                    metric_for_best_model='accuracy',
                                    load_best_model_at_end=True)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=custom_collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    train_result = trainer.train()
    import matplotlib.pyplot as plt
    # Save the fine-tuned model
    torch.save(model.state_dict(), 'finetuned_model.pth')
    # Load the fine-tuned model for evaluation
    model.load_state_dict(torch.load('finetuned_model.pth', map_location=device))
    model.eval()
    # Plot the loss function for training and evaluation data
    train_loss = train_result.training_loss
    eval_loss = trainer.evaluate().get('eval_loss')
    # Save the training loss to a file

    plt.figure(figsize=(10, 5))
    plt.plot(train_result.global_step, train_loss, label='Training Loss')
    plt.plot(train_result.global_step, eval_loss, label='Evaluation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.show()
    print("Fine-tuning complete.")


    predictions = trainer.predict(val_dataset) 
    predlabels = predictions.predictions.argmax(axis=-1)
    print(predictions.metrics)
    print('END')
