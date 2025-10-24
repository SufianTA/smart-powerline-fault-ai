import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def generate_signal(length=1024, fault=False):
    t = np.linspace(0, 1, length)
    # baseline vibration + noise
    signal = 0.2*np.sin(2*np.pi*50*t) + 0.1*np.sin(2*np.pi*120*t)
    signal += 0.05*np.random.randn(length)
    if fault:
        # inject burst / transient and/or frequency-localized energy
        center = np.random.randint(int(0.2*length), int(0.8*length))
        width = np.random.randint(10, 80)
        amplitude = np.random.uniform(0.7, 1.5)
        window = np.exp(-0.5*((np.arange(length)-center)/width)**2)
        signal += amplitude * window * np.sin(2*np.pi*np.random.randint(200, 400)*t)
        # occasional spike
        for _ in range(np.random.randint(1, 4)):
            i = np.random.randint(0, length)
            signal[i:i+3] += np.random.uniform(1.0, 2.0)
    return signal.astype(np.float32)

def generate_image(size=64, fault=False):
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    # draw the power line as a diagonal-ish line
    y0 = size//3 + np.random.randint(-3, 3)
    y1 = 2*size//3 + np.random.randint(-3, 3)
    for x in range(size):
        y = int(y0 + (y1 - y0) * (x / size))
        draw.point((x, y), fill=180)
        if np.random.rand() < 0.25:
            draw.point((x, y+1), fill=150)
    # poles
    draw.rectangle([5, size-20, 8, size-1], fill=120)
    draw.rectangle([size-10, size-22, size-7, size-1], fill=120)

    if fault:
        # bright arc / hotspot near the line
        cx = np.random.randint(size//4, 3*size//4)
        cy = y0 + (y1 - y0) * (cx / size) + np.random.randint(-2, 2)
        r = np.random.randint(3, 7)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    else:
        # slight blur / noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='data/synthetic')
    ap.add_argument('--n', type=int, default=1000, help='total samples')
    ap.add_argument('--signal_length', type=int, default=1024)
    ap.add_argument('--image_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, 'sample_images')
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in tqdm(range(args.n), desc='Generating synthetic samples'):
        label = np.random.randint(0, 2)  # 0 = normal, 1 = fault
        sig = generate_signal(args.signal_length, fault=bool(label))
        sig_path = os.path.join(args.out_dir, f'signal_{i:05d}.npy')
        np.save(sig_path, sig)

        img = generate_image(args.image_size, fault=bool(label))
        img_path = os.path.join(img_dir, f'img_{i:05d}.png')
        img.save(img_path)

        rows.append({'id': i, 'signal_path': sig_path, 'image_path': img_path, 'label': int(label)})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'sample_signals.csv')
    df.to_csv(csv_path, index=False)
    print(f'Wrote: {csv_path} and {len(rows)} images to {img_dir}')

if __name__ == '__main__':
    main()
