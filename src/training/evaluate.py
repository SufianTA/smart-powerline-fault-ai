import os, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from .utils import device, set_seed
from ..data.dataset import PowerlineDataset
from ..models.multimodal_fusion import FusionModel

def evaluate(config, ckpt):
    set_seed(config.get('seed', 42))

    ds = PowerlineDataset(
        csv_path=config['data']['signal_csv'],
        transform_img=transforms.Compose([transforms.Resize((config['data']['image_size'], config['data']['image_size'])), transforms.ToTensor()]),
        signal_length=config['data']['signal_length']
    )
    n = len(ds)
    n_train = int(n * config['data']['train_split'])
    n_val = int(n * config['data']['val_split'])
    n_test = n - n_train - n_val
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = FusionModel(
        signal_ch=config['model']['signal_channels'],
        image_ch=config['model']['image_channels'],
        hidden=config['model']['fusion_hidden'],
        dropout=config['model']['dropout']
    ).to(device())
    state = torch.load(ckpt, map_location=device())
    model.load_state_dict(state['model_state'])
    model.eval()

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for signal, image, label in loader:
            signal, image = signal.to(device()), image.to(device())
            logits = model(signal, image)
            probs = torch.softmax(logits, dim=1)[:,1]
            y_score.extend(probs.detach().cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            y_true.extend(label.numpy().tolist())

    print(classification_report(y_true, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f'ROC AUC: {auc:.4f}')
    except Exception as e:
        print('AUC unavailable:', e)
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/config.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.ckpt)

if __name__ == '__main__':
    main()
