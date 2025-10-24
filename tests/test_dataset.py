import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
# Basic smoke test: ensure dataset loads one sample
def test_dataset_smoke():
    csv = 'data/synthetic/sample_signals.csv'
    assert os.path.exists(csv), "Generate synthetic data first."
    import src.data.dataset as D
    ds = D.PowerlineDataset(csv)
    s, i, y = ds[0]
    assert s.shape[0] == 1 and len(s.shape)==3, "Signal shape (1,L) expected with batch add later."
    assert i.shape[0] == 1 and i.shape[1] > 0, "Image should be 1xHxW"
