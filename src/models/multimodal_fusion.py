import torch
import torch.nn as nn
from .cnn1d import CNN1D
from .cnn2d import CNN2D

class FusionModel(nn.Module):
    def __init__(self, signal_ch=1, image_ch=1, hidden=64, dropout=0.2):
        super().__init__()
        self.signal_encoder = CNN1D(in_ch=signal_ch)
        self.image_encoder = CNN2D(in_ch=image_ch)
        self.classifier = nn.Sequential(
            nn.Linear(64+64, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )

    def forward(self, signal, image):
        hs = self.signal_encoder(signal)
        hi = self.image_encoder(image)
        h = torch.cat([hs, hi], dim=1)
        logits = self.classifier(h)
        return logits
