"""
Model architecture for Facial Emotion Recognition.

IMPORTANT: This architecture must match EXACTLY the one used during training
(see notebook cells 22 & 25: "CELL 4 - CBAM Attention" and "CELL 7 - Model"),
otherwise the saved weights (.pth) will fail to load or load incorrectly.
"""

import torch
import torch.nn as nn
from torchvision import models

# Emotion classes in the SAME alphabetical order used by
# sorted(os.listdir(train_path)) during training (FER+ dataset).
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emoji shown in the UI next to each label (purely cosmetic)
EMOTION_EMOJI = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲',
}


class ChannelAttention(nn.Module):
    """Channel Attention part of CBAM — learns WHICH feature channels matter."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        max_ = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + max_).view(b, c, 1, 1).expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention part of CBAM — learns WHERE to look (eyes, mouth...)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


def create_model(num_classes=7):
    """Builds EfficientNet-B3 + CBAM with the same classifier head used in training."""
    model = models.efficientnet_b3(weights=None)  # weights loaded from checkpoint, not ImageNet
    model.features = nn.Sequential(model.features, CBAM(1536))
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(checkpoint_path, device='cpu', num_classes=7):
    """Creates the model and loads trained weights from a .pth checkpoint."""
    model = create_model(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Some checkpoints save {'model_state_dict': ...} instead of the raw state_dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
