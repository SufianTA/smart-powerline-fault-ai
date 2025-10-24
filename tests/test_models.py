import torch
from src.models.cnn1d import CNN1D
from src.models.cnn2d import CNN2D
from src.models.multimodal_fusion import FusionModel

def test_cnn1d():
    m = CNN1D(1)
    x = torch.randn(2,1,1024)
    y = m(x)
    assert y.shape == (2,64)

def test_cnn2d():
    m = CNN2D(1)
    x = torch.randn(2,1,64,64)
    y = m(x)
    assert y.shape == (2,64)

def test_fusion():
    m = FusionModel(1,1,hidden=32)
    s = torch.randn(2,1,1024)
    im = torch.randn(2,1,64,64)
    y = m(s, im)
    assert y.shape == (2,2)
