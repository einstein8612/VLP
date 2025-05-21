import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json
import pandas as pd

def calculate_decay_constant(r90_hrs: float):
    """
    Calculate decay constant for use in TM-21 model.

    Parameters:
    - r90_hrs: Time in hours when brightness drops to 90%.
    """

    r"""
    Convert half-life to decay constant k using:
    $$
        L(r90\_hours) = L_0 * e^{-k \cdot r90\_hours} = 0.9L_0
    $$
    $$
        e^{-k \cdot r90\_hours} = 0.9 = e^{\ln(0.9)}
    $$
    $$
        0.9 = e^{-k \cdot r90\_hours} = e^{\ln(0.9)}
    $$
    $$
        k = \frac{\ln(0.9)}{r90\_hours}
    $$
    """
    decay_k = np.log(1/0.9) / r90_hrs
    return decay_k

def calculate_relative_decay(decay_k: float, start_hours: float, end_hours: float):
    """
    Calculate the relative decay between two time points.
    This is used to calculate the relative brightness at a given time.
    
    Parameters:
    - decay_k: Decay constant.
    - start_hours: Start time in hours.
    - end_hours: End time in hours.
    """
    # np.exp(-decay_k * end_hours) / np.exp(-decay_k * start_hours) but optimised
    return np.exp(decay_k*(start_hours-end_hours))

def main():
    parser = argparse.ArgumentParser("dataset_led_age_series")
    parser.add_argument(
        "--src", help="Heatmap rank-3 tensor to simulate aging on", type=str, default="dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/heatmaps/heatmap_176_aged"
    )
    parser.add_argument(
        "--std", help="Standard deviation of noise", type=float, default=0.005
    )
    parser.add_argument(
        "--time", help="Time in hours to age LEDs", type=float, default=50000
    )
    parser.add_argument(
        "--samples_per_timestep", help="Number of noisy samples per LED per timestep", type=int, default=100
    )
    parser.add_argument(
        "--timestep", help="Timestep size", type=int, required=True, default=1000
    )
    parser.add_argument(
        "--flickering_prob", help="Probability of flickering", type=float, default=0.001
    )
    parser.add_argument(
        "--imgs", help="Whether to create images", type=bool, default=False
    )
    parser.add_argument(
        "--seed", help="Random seed", type=int, default=42
    )
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    data = np.load(args.src)
    H, W, leds_n = data.shape

    flat_data = data.reshape(W*H, leds_n)

    valid_mask = np.ones_like(data[:, :, 0], dtype=float) # Shape (H, W), 1 for valid, 0 for invalid same for all LEDs
    valid_mask[data[:, :, 0] == -1] = 0

    # Generate time steps
    timesteps = np.arange(0, args.time, args.timestep)

    # Calculate decay constant for each LED
    decay_ks = calculate_decay_constant(rng.integers(15000, 30000, leds_n))
    # Calculate relative decay for each LED at each time step
    relative_decay = np.exp(-np.outer(timesteps, decay_ks))
    # Add noise to the data
    relative_decay += rng.normal(0, args.std, size=relative_decay.shape)

    # Generate LED ids
    led_ids = np.arange(leds_n)[None, None, :] # Shape (1, 1, leds_n)
    # Generate random sample indices at each time step
    valid_flat_idxs = np.flatnonzero(valid_mask)

    sample_flat_idxs = rng.choice(valid_flat_idxs, size=(timesteps.shape[0], args.samples_per_timestep))

    # Calculate the x and y coordinates of the samples
    ys = sample_flat_idxs // W
    xs = sample_flat_idxs % W
    sample_locs = np.stack((xs, ys), axis=-1)

    # Fetch the samples for each LED at each time step
    sample_flat_idxs = sample_flat_idxs[:, :, None] # Add a new axis to the sample_flat_idxs to match the shape of led_ids
    sample_flat_idxs = np.broadcast_to(sample_flat_idxs, (timesteps.shape[0], args.samples_per_timestep, leds_n))
    led_ids = np.broadcast_to(led_ids, (timesteps.shape[0], args.samples_per_timestep, leds_n))

    # Age the samples
    samples = flat_data[sample_flat_idxs, led_ids] # Get the samples for each LED at each timestep with the same sample index. Shape (timesteps, samples_per_timestep, leds_n)
    aged_samples = samples * relative_decay[:, None, :] # Apply the relative decay to the samples
    
    flickering = rng.choice([0, 1], size=aged_samples.shape, p=[args.flickering_prob, 1 - args.flickering_prob])
    aged_samples = aged_samples * flickering # Apply flickering to the samples
    
    print(f"Simulated {(1-flickering).sum()} LEDs flickering")

     # Reshape to (timesteps * samples_per_timestep, leds_n), C-style row-major order so that the first axis is the slowest changing axis
    reshaped = aged_samples.reshape(-1, samples.shape[2]) # Shape (timesteps * samples_per_timestep, leds_n)
    sample_locs = sample_locs.reshape(-1, 2) # Shape (timesteps * samples_per_timestep, 2)

    os.makedirs(args.dst, exist_ok=True)

    df = pd.DataFrame(
        data=np.hstack((sample_locs*10, reshaped)),
        columns=['x', 'y'] + [f'led_{i}' for i in range(reshaped.shape[1])],
    ).astype({
        'x': 'int32',
        'y': 'int32',
    })
    
    print("Aging data...")
    # Save the dataframe to a CSV file
    df.to_csv(args.dst + "/aged_samples.csv", index=False)
    
    # Save the data to a json
    led_decays = {f"LED {i}": decay_k for i, decay_k in enumerate(decay_ks)}
    json.dump(led_decays, open(args.dst + "/decay_ks.json", "w"))

    if not args.imgs:
        return

    aged_data = data.copy() * relative_decay[-1, :] # Apply the relative decay to the data
    aged_data = np.clip(aged_data, 0, None) # Clip negative values to 0
    for i in tqdm(range(leds_n), "Exporting heat maps for aged data"):
        plt.imshow(aged_data[:, :, i], interpolation='nearest', origin='lower', vmin=0, vmax=data[:, :, i].max())
        plt.colorbar()
        plt.title(f"Aged Data for LED {i} ({args.time} Hours)\n(decay_k = {decay_ks[i]})")
        plt.savefig(args.dst + f"/led_{i}_aged_{args.time}_hours.png")
        plt.clf()

if __name__ == "__main__":
    main()