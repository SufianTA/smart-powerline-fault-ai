import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class PowerlineDataset(Dataset):
    def __init__(self, csv_path, transform_img=None, signal_length=1024):
        self.df = pd.read_csv(csv_path)
        self.transform_img = transform_img
        self.signal_length = signal_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = np.load(row['signal_path']).astype(np.float32)
        if len(signal) != self.signal_length:
            # pad / trim
            if len(signal) > self.signal_length:
                signal = signal[:self.signal_length]
            else:
                pad = self.signal_length - len(signal)
                signal = np.pad(signal, (0, pad), mode='constant')
        signal = torch.from_numpy(signal)[None, :]  # (1, L)

        img = Image.open(row['image_path']).convert('L')
        if self.transform_img:
            img = self.transform_img(img)
        else:
            import torchvision.transforms as T
            img = T.Compose([T.Resize((64,64)), T.ToTensor()])(img)

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return signal, img, label
