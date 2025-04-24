# VLP

# Prerequisites

This project was built with the following software installed:
- Python 3.13

# Get Started

In order to get started we need to install the requirements

## Create virtual environment (Optional)
```bash
python -m venv .venv
source .venv/scripts/activate # .\.venv\Scripts\Activate.ps1 on Windows
```

## Install requirements
```bash
pip install -r requirements.txt
```

## Data

To get started aqcuire the `mat_files` folder and store them in the dataset folder. Afterwards, run the script to translate them into a PyTorch set

```bash
python ./dataset/convert.py --src "./dataset/mat_files" --dst "./dataset/data.csv" --normalise true
```