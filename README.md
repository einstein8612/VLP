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

This is also a necessary step if you want to clean or augment the data.

```bash
$ python dataset/heatmap.py --src "./dataset/exported/data.csv" --dst "./dataset/heatmaps" --imgs true
```

An example of such a heatmap is given here, for the non-cleaned data of LED 16 at `z=176`.

![LED 16 Heatmap](./assets/readme/led_16_heatmap.png)

## Clean data

Afterwards, you can clean the data generated in the heatmaps with any of the following four, at the time of writing, strategies:

- MEAN
- IDW (Inverse Distance Weighing)
- LAMBERTIAN
- LAMBERTIAN-IDW (Inverse Distance Weighing)

It should be clear what they do: they replace invalid or noisy points using different methods. These include the mean of the nearest valid points, the inverse distance weighted sum of the nearest valid points, an estimated RSS based on the Lambertian model of the closest valid point, and an estimated RSS using the Lambertian model for the nearest valid points, weighted by their inverse distance.

In order to run them use the following command:

```bash
python dataset/clean.py --src {SRC} --dst {DST} --strategy {STRATEGY} --imgs {IMGS}
```

### Example (Lambertian IDW)

```bash
python dataset/clean.py --src "dataset/heatmaps/heatmap_176/raw.npy" --dst "dataset/heatmaps/heatmap_176" --strategy LAMBERTIAN-IDW --imgs true
```

Afterwards, your heatmaps will have been cleaned and images are stored in the destination folder under the name `led_{i}_cleaned_{STRATEGY}.png`

Again, an example of such a heatmap is given here for LED 16.

![LED 16 Cleaned Heatmap](./assets/readme/led_16_cleaned_heatmap.png)

**Note**: In the actual output images, the raw version will not be plotted. This is simply to showcase the cleaning process, the actual output will be just the right sub-figure.

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

### Example (MLP)

```bash
$ python experiment.py --task "MLP" --dataset "./dataset/exported/data_176" --device "cuda:0" --seed 42
# ...
# Average error: 6.804797172546387
```