import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.out = nn.Identity()

    def forward(self, x):
        # x: (B,1,L)
        h = self.net(x)  # (B,64,1)
        h = h.squeeze(-1) # (B,64)
        return h
