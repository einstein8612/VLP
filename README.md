# VLP

# Prerequisites

This project was built with the following software installed:
- Python 3.13

# Get Started

In order to get started we need to install the requirements

## Create virtual environment (Optional)
```bash
$ python -m venv .venv
source .venv/bin/activate # .\.venv\Scripts\Activate.ps1 on Windows
```

## Install requirements
```bash
$ pip install -r requirements.txt
```

## Data

To get started aqcuire the `mat_files` folder and store them in the dataset folder. Afterwards, run the script to translate them into a PyTorch set

```bash
$ python dataset/convert.py --src "./dataset/mat_files" --dst "./dataset/exported" --normalise true --training_fraction 0.8 --seed 42
```

This will generate the files

- `{dst}/data_{z}/train.csv`
- `{dst}/data_{z}/test.csv`
- `{dst}/data.csv`

In the given destination folder, where the first two are a split of the data grouped by z-pos, and the last is all the data in one CSV.

## Generate heat maps

In order to understand the quality of your cleaning solution or data augmentation solution, you can generate a heatmaps for every LED. Here every heatmap corresponds to the mean of all the values that are associated with that specific (x,y) coordinate.

```bash
$ python dataset/heatmap.py --src "./dataset/exported/data.csv" --dst "./dataset/heatmaps"
```

An example of such a heatmap is given here, for the non-cleaned data of LED 16 at `z=176`.

![LED 16 Heatmap](./assets/readme/led_16_heatmap.png)

## Run experiment (Training)

In order to test this data's efficiency at generating positions, we can run experiments with the following command:

```bash
$ python experiment.py --task {TASK} --dataset {DATASET} --seed {SEED}
```

### Example (RF)

```bash
$ python experiment.py --task "RF-TINY" --dataset "./dataset/exported/data_176" --seed 42
# ...
# Model saved to saved_runs/RF-TINY-1745593383.pickle
# Average error: 24.283416141929006
```

## (Re?)Run experiment (From trained model)

In order to be repeatable, we can used the saved run to predict the average error again.

```bash
$ python experiment.py --task {TASK} --dataset {DATASET} --load {SAVED_RUN} --seed {SEED}
```

### Example (RF)

```bash
$ python experiment.py --task "RF-TINY" --dataset "./dataset/exported/data_176" --load "./saved_runs/RF-TINY-1745593383.pickle" --seed 42
# ...
# Average error: 24.283416141929006
```

## CUDA/ROCm

If you want to use your GPU to accelerate training, then pass the device parameter as follows:

```bash
$ python experiment.py --task {TASK} --dataset {DATASET} --device {DEVICE} --seed {SEED}
```

### Example (RF)

```bash
$ python experiment.py --task "MLP" --dataset "./dataset/exported/data_176" --device "cuda:0" --seed 42
# ...
# Average error: 6.804797172546387
```