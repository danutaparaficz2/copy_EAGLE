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

if __name__ == '__main__':
    # Usage
    path = "/Users/danuta.paraficz/PycharmProjects/eagle-classification/Data/Duramat_no_pool_labels.pkl"
    data_loader =  Load_Data(path)
    data = data_loader.get_data()

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_dataset = PVDataset(train_data, channels=[0, 1, 2], transform=transform, scale=1)
    val_dataset = PVDataset(val_data, channels=[0, 1, 2], transform=transform, scale=1)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=15, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=15, shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Load the model
    model = so2sat_channelvit_small_p8_with_hcs_random_split_supervised(pretrained=False)

    # Load the pretrained weights and map them to the appropriate device
    state_dict = torch.load('Data/so2sat_channelvit_small_p8_with_hcs_hard_split_supervised.pth', map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model.to(device)

    # Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # data augmontation part? rotation etc!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Training loop
    num_epochs = 5
    model.train()
    print(f"Size of training data: {len(train_data)}")
   
    for epoch in range(num_epochs):
        k=0
        for data in train_dataloader:
            data = {k: v.to(device) for k, v in data.items()}
          #  keys_to_select = ['channels', 'labels']

            # Create a new dictionary with only the selected keys
           # selected_dict = {key: data[key] for key in keys_to_select if key in data}
          #  print(selected_dict)  # Output: {'a': 1, 'c': 3}
            # Forward pass
            outputs = model(data["images"], extra_tokens=data)
            loss = criterion(outputs, data['labels'])
            k = k+15
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.mps.empty_cache()
            print(epoch, k)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        torch.save(model.state_dict(), 'finetuned_model'+str(epoch)+'.pth')

    print("Fine-tuning complete.")
    # Save the fine-tuned model
    torch.save(model.state_dict(), 'finetuned_model.pth')



    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dataloader:
            data =  {k: v.to(device) for k, v in data.items()}
            outputs = model(data, extra_tokens=data)
            _, predicted = torch.max(outputs.data, 1)
            total += data['labels'].size(0)
            correct += (predicted == data['labels']).sum().item()
            print(correct)

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    print('END')
