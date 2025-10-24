# Smart Fault Detection in Power Lines Using AI and Optical Sensors

End‑to‑end, multimodal reference project that detects power‑line faults using **optical sensor time‑series** (e.g., Distributed Acoustic Sensing — DAS) and **camera imagery**.  
It includes synthetic data generation, model training, evaluation, and an interactive Streamlit app for inference.

> Shareable with students (JHU‑ready): fully self‑contained, no external datasets, with tests and docs.

---

## Features

- **Synthetic data generator** for both modalities (signals + images) with configurable noise/fault patterns.
- **Multimodal PyTorch models**:
  - 1D CNN for optical sensor time‑series.
  - 2D CNN for camera images.
  - Late‑fusion head to combine modalities.
- **Training & evaluation** with metrics, confusion matrices, and ROC.
- **Streamlit demo app** for quick what‑if tests.
- **Config‑driven** (`configs/config.yaml`) and easily extensible.
- **Unit tests** for datasets and models.
- **Reproducible**: requirements pinned, deterministic seeding.

---

## Quickstart

```bash
# 1) Create & activate a virtualenv (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Generate synthetic data
python -m src.data.generate_synthetic_data --out_dir data/synthetic --n 1200

# 4) Train the model (multimodal by default)
python -m src.training.train --config configs/config.yaml

# 5) Evaluate on the test split
python -m src.training.evaluate --config configs/config.yaml --ckpt runs/best_model.pt

# 6) Run the Streamlit app
streamlit run app/streamlit_app.py
```

> Trained checkpoints are saved into `runs/`. You can point the app/evaluator to any checkpoint using the `--ckpt` flag or the config file.

---

## Repository Structure

```
smart-powerline-fault-ai/
├─ app/
│  └─ streamlit_app.py
├─ configs/
│  └─ config.yaml
├─ data/
│  └─ synthetic/              # generated via script
├─ notebooks/
│  └─ 01_exploration.ipynb
├─ scripts/
│  ├─ run_all.sh
│  └─ run_all.bat
├─ src/
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ dataset.py
│  │  └─ generate_synthetic_data.py
│  ├─ features/
│  │  └─ feature_engineering.py
│  ├─ inference/
│  │  └─ infer.py
│  ├─ models/
│  │  ├─ cnn1d.py
│  │  ├─ cnn2d.py
│  │  └─ multimodal_fusion.py
│  ├─ training/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ utils.py
│  └─ visualization/
│     └─ plot_signals.py
├─ tests/
│  ├─ test_dataset.py
│  └─ test_models.py
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

---

## Research Notes & Extensions

This project simulates a realistic scenario:

- **Time‑series** mimics DAS backscatter changes along fiber near power lines. Faults produce impulsive, bursty, or frequency‑localized energy.
- **Images** mimic camera snapshots of lines/insulators; faults introduce small geometric/texture anomalies (hotspots/arc‑like blobs).

Extensions to explore:
- Replace synthetic generator with real datasets (e.g., utility DAS, line patrol images).
- Add **temporal context** with 1D CNN + BiLSTM.
- Try **self‑supervised pretraining** (SimCLR/MoCo) on unlabeled signals/images.
- Deploy the Streamlit app via container on Render/Cloud Run with GPU support.

---

## License

Apache 2.0 — see `LICENSE`.
