import torch, yaml
from PIL import Image
import numpy as np
from ..models.multimodal_fusion import FusionModel
from ..training.utils import device
import torchvision.transforms as T

def load_model(ckpt_path):
    state = torch.load(ckpt_path, map_location=device())
    cfg = state['config']
    model = FusionModel(
        signal_ch=cfg['model']['signal_channels'],
        image_ch=cfg['model']['image_channels'],
        hidden=cfg['model']['fusion_hidden'],
        dropout=cfg['model']['dropout']
    ).to(device())
    model.load_state_dict(state['model_state'])
    model.eval()
    return model, cfg

def predict(ckpt_path, signal_path, image_path):
    model, cfg = load_model(ckpt_path)
    signal = np.load(signal_path).astype(np.float32)
    L = cfg['data']['signal_length']
    if len(signal) != L:
        if len(signal) > L:
            signal = signal[:L]
        else:
            signal = np.pad(signal, (0, L - len(signal)), mode='constant')
    signal = torch.from_numpy(signal)[None,None,:].to(device())

    img = Image.open(image_path).convert('L')
    tf = T.Compose([T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])), T.ToTensor()])
    img = tf(img)[None,:].to(device())

    with torch.no_grad():
        logits = model(signal, img)
        prob_fault = torch.softmax(logits, dim=1)[0,1].item()
        pred = int(logits.argmax(dim=1).item())
    return {'pred_label': pred, 'prob_fault': prob_fault}

def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--signal', required=True)
    ap.add_argument('--image', required=True)
    args = ap.parse_args()
    res = predict(args.ckpt, args.signal, args.image)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
