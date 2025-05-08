import argparse
import os

import numpy as np
from tqdm import tqdm

import pandas as pd

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser("dataset_heatmap")
    parser.add_argument(
        "--src", help="File to import from", type=str, default="dataset/exported/data.csv"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/heatmaps"
    )
    parser.add_argument(
        "--imgs", help="Whether to create images", type=bool, default=False
    )
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    df = pd.read_csv(args.src)
    
    # Change x and y into matrix coordinates
    df["x"] = df["x"]-df["x"].min()
    df["y"] = df["y"]-df["y"].min()
    
    df["x"] = df["x"] / 10
    df["y"] = df["y"] / 10

    # Fix types
    df = df.astype({"x": np.int16, "y": np.int16})

    # Add iteration column to distinguish between different measurements
    df["iter"] = df.groupby(["x", "y", "z"]).cumcount()
    no_iter = df["iter"].max() + 1

    for z in df["z"].unique():
        data_z = df[df["z"] == z]
        data_z = data_z.drop(columns=["z"])

        os.makedirs(args.dst + f"/heatmap_{z}", exist_ok=True) # Make directory for each z

        # Construct matrix for heatmaps
        y_size = data_z["y"].max()+1
        x_size = data_z["x"].max()+1
        leds = data_z.columns.difference(["x","y","iter"])
        leds = sorted(leds, key=lambda x: int(x.split('_')[1]))
        led_size = len(leds)
        
        matrix = -np.ones((y_size, x_size, led_size, no_iter)) # Initialize with -1 to indicate no data
        for i, led in enumerate(leds):
            matrix[df["y"],df["x"], i, df["iter"]] = df[led]

        np.save(args.dst+f"/heatmap_{z}/raw.npy", matrix)

        if not args.imgs:
            continue

        matrix = np.mean(matrix, axis=3) # Create images using mean
        matrix = np.clip(matrix, 0, None) # Clip negative values to 0

        for i in tqdm(range(matrix.shape[2]), f"Exporting heat maps for z={z}"):
            plt.imshow(matrix[:, :, i], interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title(f"Raw Data for LED {i}")
            plt.savefig(args.dst+f"/heatmap_{z}/led_{i}.png")
            plt.clf()

    print(f"Exported heatmaps to {args.dst}")

if __name__ == "__main__":
    main()