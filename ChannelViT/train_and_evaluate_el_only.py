"""
Train and evaluate the multimodal model using only EL data (no VIS/UV) for both training and evaluation.
"""
import os
import pickle
import torch
from sklearn.metrics import classification_report
from Ai_try.multibranch_fusion import MultiBranchFusion
from safetensors.torch import load_file
from main_multimodal import prepare_el_only_data

# Load filtered data
filtered_data_path = os.path.join('Data/images', "filtered_data.pkl")
with open(filtered_data_path, "rb") as f:
    filtered_data = pickle.load(f)

# Prepare EL-only train/val splits
el_train, el_val = prepare_el_only_data(
    filtered_data["data_Duramat_filtered_more"],
    filtered_data["data_Infinity_filtered_more"]
)

# Train model (Phase A only, fusion head)
model = MultiBranchFusion(num_classes=7)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    for sample in el_train:
        if isinstance(sample, dict):
            img = sample['image']
            label = torch.tensor(sample['label'], dtype=torch.float32)
        else:
            img = sample[0]
            label = torch.tensor(sample[1], dtype=torch.float32)
        optimizer.zero_grad()
        logits = model(el=img.unsqueeze(0))['logits'].squeeze(0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(el_train):.4f}")

# Save model
os.makedirs('./el_only_output', exist_ok=True)
torch.save(model.state_dict(), './el_only_output/model.pt')

# Evaluate on EL-only validation set
model.eval()
y_true = []
y_pred = []
for sample in el_val:
    if isinstance(sample, dict):
        img = sample['image']
        label = sample['label']
    else:
        img = sample[0]
        label = sample[1]
    with torch.no_grad():
        logits = model(el=img.unsqueeze(0))['logits']
        pred = logits.argmax(dim=-1).item()
    y_true.append(label.index(1) if 1 in label else 0)
    y_pred.append(pred)

print("\nEL-only evaluation (no VIS/UV):")
print(classification_report(y_true, y_pred, digits=3))
