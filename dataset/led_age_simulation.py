import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json

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
    parser = argparse.ArgumentParser("dataset_led_age_simulation")
    parser.add_argument(
        "--src", help="Heatmap rank-3 tensor to simulate aging on", type=str, default="dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/heatmaps/heatmap_176_aged"
    )
    parser.add_argument(
        "--std", help="Standard deviation of noise", type=float, default=0.1
    )
    parser.add_argument(
        "--min_age", help="Minimum aging done to LEDs in hours", type=float, default=0
    )
    parser.add_argument(
        "--max_age", help="Maximum aging done to LEDs in hours", type=float, default=100000
    )
    parser.add_argument(
        "--ages", help="Aging done to LEDs in hours", nargs="*", type=float, required=False
    )
    parser.add_argument(
        "--r90_hours", help="Time in hours until 90%% of received signal strength is left. Taken from TM-80 report or similar.", type=float, default=33000
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
    leds_n = data.shape[2]
    
    if args.ages is None:
        args.ages = rng.integers(args.min_age, args.max_age, leds_n)
    else:
        args.ages = np.array(args.ages)
    
    if (len(args.ages) != data.shape[2]):
        raise ValueError(f"Number of ages ({len(args.ages)}) does not match number of LEDs ({data.shape[2]}).")
    
    if np.min(args.ages) < 0:
        raise ValueError(f"Minimum age ({np.min(args.ages)}) cannot be negative.")

    decay_k = calculate_decay_constant(args.r90_hours)

    relative_decay = calculate_relative_decay(decay_k, 0, args.ages)
    
    # Apply the relative decay to the data
    aged_data = data * relative_decay
    
    led_ages = {f"LED {i}": int(age) for i, age in enumerate(args.ages)}
    
    print("Aging data with the following hours:")
    print("\n".join([f"{name}: {age} hours" for name, age in led_ages.items()]))

    os.makedirs(args.dst, exist_ok=True)

    np.save(args.dst + "/aged.npy", aged_data)
    json.dump(led_ages, open(args.dst + "/ages.json", "w"))

    if not args.imgs:
        return

    aged_data = np.clip(aged_data, 0, None) # Clip negative values to 0
    for i in tqdm(range(leds_n), "Exporting heat maps for aged data"):
        plt.imshow(aged_data[:, :, i], interpolation='nearest', origin='lower', vmin=0, vmax=data[:, :, i].max())
        plt.colorbar()
        plt.title(f"Aged Data for LED {i} ({args.ages[i]} Hours)")
        plt.savefig(args.dst + f"/led_{i}_aged_{args.ages[i]}_hours.png")
        plt.clf()

if __name__ == "__main__":
    main()