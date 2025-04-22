import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalSolarCellClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        # Load pre-trained backbones (using smaller ones for small data)
        self.encoder_el = models.resnet18(pretrained=pretrained)
        self.encoder_vis = models.resnet18(pretrained=pretrained)
        self.encoder_uv = models.resnet18(pretrained=pretrained)

        # Modify backbones to output features (remove final FC layer)
        self.num_features_el = self.encoder_el.fc.in_features
        self.num_features_vis = self.encoder_vis.fc.in_features
        self.num_features_uv = self.encoder_uv.fc.in_features
        self.encoder_el.fc = nn.Identity()
        self.encoder_vis.fc = nn.Identity()
        self.encoder_uv.fc = nn.Identity()

        # Fusion and Classifier Head
        # Example: Simple concatenation
        fused_feature_dim = self.num_features_el + self.num_features_vis + self.num_features_uv
        self.classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # Output layer (raw logits)
        )

    def forward(self, img_el, img_vis=None, img_uv=None):
        # Extract features - always run EL
        feat_el = self.encoder_el(img_el)

        # Handle missing modalities - create zero tensors matching device and batch size
        batch_size = img_el.shape[0]
        device = img_el.device

        if img_vis is not None:
            feat_vis = self.encoder_vis(img_vis)
        else:
            feat_vis = torch.zeros(batch_size, self.num_features_vis, device=device)

        if img_uv is not None:
            feat_uv = self.encoder_uv(img_uv)
        else:
            feat_uv = torch.zeros(batch_size, self.num_features_uv, device=device)

        # Fuse features (concatenation)
        fused_features = torch.cat([feat_el, feat_vis, feat_uv], dim=1)

        # Classify
        logits = self.classifier(fused_features)
        return logits

# --- Training Loop Setup ---
# Need a custom Dataset that returns a dictionary like:
# {'el': tensor_el, 'vis': tensor_vis_or_None, 'uv': tensor_uv_or_None, 'label': label_idx}
# The training loop then unpacks this and calls model(img_el=batch['el'], img_vis=batch['vis'], img_uv=batch['uv'])