import numpy as np
import numpy.typing as npt
import pandas as pd

def heatmap_to_data_split_from_file(file_str: str, training_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(file_str)

    return heatmap_to_data_split(data, training_fraction, seed)

def heatmap_to_data_split(data: npt.NDArray, training_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    h, w, led_n = data.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((xx.ravel(), yy.ravel()), axis=-1)

    led_columns = data.reshape(h*w, led_n, order='C')
    dataset = np.concatenate((coords, led_columns), axis=1)

    df = pd.DataFrame(dataset, columns=['x', 'y'] + [f'led_{i}' for i in range(led_columns.shape[1])])
    df = df.astype({"x": np.int16, "y": np.int16})

    # Scale x and y to match the original coordinates
    df["x"] = df["x"]*10
    df["y"] = df["y"]*10

    # Remove all rows with -1 values
    df = df[~df.isin([-1]).any(axis=1)]

    data_train = df.sample(frac=training_fraction, random_state=seed)
    data_test = df.drop(data_train.index).sample(frac=1.0, random_state=seed) # Shuffle

    return data_train, data_test

