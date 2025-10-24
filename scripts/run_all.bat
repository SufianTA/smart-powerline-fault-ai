@echo off
python -m src.data.generate_synthetic_data --out_dir data/synthetic --n 1200
python -m src.training.train --config configs\config.yaml
python -m src.training.evaluate --config configs\config.yaml --ckpt runs\best_model.pt
