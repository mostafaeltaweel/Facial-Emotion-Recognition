import torch
import torch.nn as nn
from torchvision import models

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class ChannelAttention(nn.Module):
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
        batch, channels, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(batch, channels))
        maximum = self.fc(self.max_pool(x).view(batch, channels))
        attention = self.sigmoid(avg + maximum).view(batch, channels, 1, 1)
        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


def create_model(num_classes=7):
    model = models.efficientnet_b3(weights=None)
    model.features = nn.Sequential(model.features, CBAM(1536))
    features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(features, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Dropout(0.4), nn.Linear(512, num_classes),
    )
    return model


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # Supports checkpoints saved from DataParallel, too.
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model = create_model()
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
