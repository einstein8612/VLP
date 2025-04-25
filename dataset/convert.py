import argparse
import os

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat
from tqdm import tqdm

import pandas as pd

def load_data(folder_path: str, normalise: bool) -> npt.NDArray[np.float32]:
    """
    Load data from a folder containing .mat files.
    """
    # Get all relevant .mat files in the folder
    mat_files = [f for f in os.listdir(folder_path) if "row" in f]

    # Load each .mat file and concatenate the data
    data = []
    for file in tqdm(mat_files, "Loading .mat files"):
        file_path = os.path.join(folder_path, file)
        data.append(load_data_from_mat(file_path))

    data = np.concatenate(data, axis=0)

    # Normalise the data
    if normalise:
        # Normalise the data to the range [0, 1]
        max_val = np.max(data[:, 3:], axis=0)
        data[:, 3:] = data[:, 3:] / max_val
    
    return data


def load_data_from_mat(file_path: str) -> npt.NDArray[np.float32]:
    """
    Load data from a single .mat file.
    """
    # Load the .mat file
    mat = loadmat(file_path)

    rx_id = mat["rx_id"][0] - 1  # Zero-based index
    no_it = mat["no_it"][0][0].astype(np.int16)
    offset = mat["offset"]  # Offset is the offset of each receiver
    resolution = mat["resolution"][0][0].astype(np.int16)
    pos_z = mat["height"][0][0]

    # Measurement data
    channel_data = mat["swing"]

    row_id = int(file_path.split("_")[-1].split(".")[0])

    # Create a list to hold the rows
    row_amount = len(rx_id) * channel_data.shape[3] * no_it
    rows = np.zeros((row_amount, 39), dtype=np.float32)

    i = 0
    for id in rx_id:
        offset_x = offset[id][0]
        offset_y = offset[id][1]
        for x in range(0, channel_data.shape[3]):
            for it in range(0, no_it):
                # Select 1 measurement
                measurement = channel_data[:, id, it, x]

                # Calculate position of the RX for this measurement
                y = row_id
                pos_x = offset_x + y * resolution
                pos_y = offset_y + x * resolution

                pos = np.array([pos_x, pos_y, pos_z], dtype=np.float64)
                row = np.concatenate((pos, measurement), axis=0).reshape(1, -1)

                rows[i, :] = row

                i += 1

    return rows

def main():
    parser = argparse.ArgumentParser("dataset_convert")
    parser.add_argument(
        "--src", help="Folder to import from", type=str, default="dataset/mat_files"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/converted"
    )
    parser.add_argument(
        "--normalise", help="Whether to normalise data", type=bool, default=True
    )
    parser.add_argument(
        "--training_fraction", help="Fraction of training set", type=float, default=0.8
    )
    parser.add_argument(
        "--seed", help="Random seed", type=int, default=42
    )
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    data = load_data(args.src, args.normalise)

    # Load and sort the data
    df= pd.DataFrame(data, columns=["x", "y", "z"]+["led_"+str(i) for i in range(0, 36)])
    df.sort_values(by=["x", "y", "z"], inplace=True)

    # Assign the correct data types to the columns
    df = df.astype({"x": np.int16, "y": np.int16, "z": np.int16})
    df = df.astype({"led_" + str(i): np.float32 for i in range(0, 36)})

    print(f"Exporting raw data to {args.dst}/data.csv")
    df.to_csv(args.dst + "/data.csv", index=False)

    for z in tqdm(df["z"].unique(), "Exporting data splits"):
        data_z = df[df["z"] == z]
        data_z = data_z.drop(columns=["z"])

        data_z_train = data_z.sample(frac=args.training_fraction, random_state=args.seed)
        data_z_test = data_z.drop(data_z_train.index).sample(frac=1.0, random_state=args.seed) # Shuffle

        os.makedirs(args.dst + f"/data_{z}", exist_ok=True) # Make directory for each z

        data_z_train.to_csv(args.dst + f"/data_{z}/train.csv", index=False)
        data_z_test.to_csv(args.dst + f"/data_{z}/test.csv", index=False)

    print(f"Exported data to {args.dst}")

if __name__ == "__main__":
    main()