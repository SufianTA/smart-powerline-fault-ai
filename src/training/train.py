import os, yaml, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .utils import set_seed, device
from ..data.dataset import PowerlineDataset
from ..models.multimodal_fusion import FusionModel

def train(config):
    set_seed(config.get('seed', 42))

    # Dataset & splits
    ds = PowerlineDataset(
        csv_path=config['data']['signal_csv'],
        transform_img=transforms.Compose([transforms.Resize((config['data']['image_size'], config['data']['image_size'])), transforms.ToTensor()]),
        signal_length=config['data']['signal_length']
    )
    n = len(ds)
    n_train = int(n * config['data']['train_split'])
    n_val = int(n * config['data']['val_split'])
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    # Model
    model = FusionModel(
        signal_ch=config['model']['signal_channels'],
        image_ch=config['model']['image_channels'],
        hidden=config['model']['fusion_hidden'],
        dropout=config['model']['dropout']
    ).to(device())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    best_val = 0.0
    out_dir = config['logging']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, config['logging']['ckpt_name'])

    for epoch in range(1, config['training']['epochs']+1):
        model.train()
        losses = []
        preds_all, labels_all = [], []

        for signal, image, label in tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]} [train]'):
            signal, image, label = signal.to(device()), image.to(device()), label.to(device())

            optimizer.zero_grad()
            logits = model(signal, image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            preds_all.extend(preds)
            labels_all.extend(label.detach().cpu().numpy().tolist())

        train_acc = accuracy_score(labels_all, preds_all)
        # Validation
        model.eval()
        v_preds, v_labels, v_losses = [], [], []
        with torch.no_grad():
            for signal, image, label in tqdm(val_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]} [val]'):
                signal, image, label = signal.to(device()), image.to(device()), label.to(device())
                logits = model(signal, image)
                v_losses.append(criterion(logits, label).item())
                v_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
                v_labels.extend(label.detach().cpu().numpy().tolist())

        val_acc = accuracy_score(v_labels, v_preds)
        print(f'Epoch {epoch}: train_loss={sum(losses)/len(losses):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')

        if val_acc >= best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'config': config}, ckpt_path)
            print(f'✅ Saved best checkpoint to {ckpt_path} (val_acc={val_acc:.4f})')

    print('Training complete.')
    return ckpt_path

def main():
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/config.yaml')
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == '__main__':
    main()
