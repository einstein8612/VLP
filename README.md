# VLP

# Prerequisites

This project was built with the following software installed:
- Python 3.13

# Get Started

In order to get started we need to install the requirements

## Create virtual environment (Optional)
```bash
$ python -m venv .venv
source .venv/scripts/activate # .\.venv\Scripts\Activate.ps1 on Windows
```

## Install requirements
```bash
$ pip install -r requirements.txt
```

## Data

To get started aqcuire the `mat_files` folder and store them in the dataset folder. Afterwards, run the script to translate them into a PyTorch set

```bash
$ python ./dataset/convert.py --src "./dataset/mat_files" --dst "./dataset/exported" --normalise true --training_fraction 0.8 --seed 42
```

This will generate the files

- `{dst}/data_{z}/train.csv`
- `{dst}/data_{z}/test.csv`
- `{dst}/data.csv`

In the given destination folder, where the first two are a split of the data grouped by z-pos, and the last is all the data in one CSV.

## Run experiment (Training)

In order to test this data's efficiency at generating positions, we can run experiments with the following command:

```bash
$ python experiment.py --task {TASK} --dataset {DATASET} --seed 42
```

### Example (RF)

```bash
$ python .\experiment.py --task "RF-TINY" --dataset "./dataset/exported/data_176" --seed 42
# ...
# Model saved to saved_runs/RF-TINY-1745593383.pickle
# Average error: 24.283416141929006
```

## (Re?)Run experiment (From trained model)

In order to be repeatable, we can used the saved run to predict the average error again.

```bash
$ python experiment.py --task {TASK} --dataset {DATASET} --load {SAVED_RUN} --seed 42
```

### Example (RF)

```bash
$ python .\experiment.py --task "RF-TINY" --dataset "./dataset/exported/data_176" --load "./saved_runs/RF-TINY-1745593383.pickle" --seed 42
# ...
# Average error: 24.283416141929006
```