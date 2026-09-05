# KGRV-former for Fine-Grained Human Values Classification

This repository contains the implementation for **KGRV-former**, a Python-based model for fine-grained human values classification.

## Overview

KGRV-former is designed to classify text into fine-grained human value categories. The model script in this repository can be used to train, evaluate, and reproduce the study setup with the environment listed below.

## Repository Structure

- `model` script(s): core implementation for training and inference
- `requirment.tex`: replication environment specification (LaTeX)

> If your main model file has a specific name (for example `train.py`, `model.py`, or `main.py`), replace mentions of “model script” below with that filename.

## System Configuration (Replication)

The following hardware/software environment was used:

- **Operating system:** Ubuntu 22.04
- **Python:** 3.12
- **PyTorch:** 2.5.1
- **CUDA:** 12.4
- **GPU:** NVIDIA RTX 4090 (24 GB)
- **CPU:** Intel Xeon Gold 6430
- **System memory:** 120 GB

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Sfahad05/KGRV-former-for-Fine-Grained-Human-values-Classifications.git
cd KGRV-former-for-Fine-Grained-Human-values-Classifications
```

2. Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
# If you provide a requirements.txt file:
pip install -r requirements.txt
```

4. Verify PyTorch + CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Reproduction Notes

- Use the exact versions listed in `requirment.tex` where possible.
- Fix random seeds for deterministic runs.
- Keep train/validation/test splits consistent with the study protocol.
- Report GPU model, CUDA, and PyTorch version in logs for reproducibility.

## Example Usage

If your entrypoint script is `main.py`, a typical run may look like:

```bash
python main.py --config config.yaml
```

If your training script is different, replace `main.py` with the actual file name and arguments.

## Citation

If this code supports a paper/preprint, please add citation details here.

## License

Please add your preferred license (e.g., MIT, Apache-2.0).
