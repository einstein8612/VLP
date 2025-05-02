import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from tqdm import tqdm
import os

score_threshold = 0.05  # Threshold for score matrix to consider a point bad

def get_tx_positions():
    ##Actual position of the TX (roughly measured)
    return [[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
            [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
            [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
            [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
            [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
            [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]]

def generate_score_matrix(data: npt.NDArray, r=1):
    # Create a mask for valid data points
    valid_mask = np.ones_like(data, dtype=float)
    valid_mask[data == -1] = 0

    # Create a kernel for convolution
    kernel_size = 2 * r + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    kernel[r, r] = 0  # exclude the center

    clipped_data = np.clip(data, 0, None)  # Ensure no negative values for convolution

    # Apply convolution
    refs_sum = convolve(clipped_data, kernel, mode='constant', cval=0.0, axes=(0, 1))
    count_neighbors = convolve(valid_mask, kernel, mode='constant', cval=0.0, axes=(0, 1))

    # We only divide by count_neighbors where it is valid
    refs_mean = np.divide(refs_sum, count_neighbors, where=valid_mask != 0)
    refs_mean[valid_mask == 0] = 0 # Set invalid points to 0

    bias = (1 / (refs_mean + 1e-6)) ** 0.25

    score_matrix = np.abs(clipped_data - refs_mean) * bias
    return score_matrix

def reconstruct_rss_lambertian(rss_ref, d1, d2, m):
    """ Reconstructs RSS at d1 using known RSS at d2 with Lambertian model. """
    exponent = m + 3
    return rss_ref * (d2 / d1) ** exponent

strategies = ["MEAN", "IDW", "LAMBERTIAN", "LAMBERTIAN-IDW"]

def main():
    parser = argparse.ArgumentParser("dataset_clean")
    parser.add_argument(
        "--src", help="Heatmap rank-3 tensor to clean", type=str, default="dataset/heatmaps/heatmap_176/raw.npy"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/heatmaps/heatmap_176"
    )
    parser.add_argument(
        "--strategy", help="Either LAMBERTIAN or MEAN", type=str, default="MEAN"
    )
    parser.add_argument(
        "--k", help="Number of neighbors to consider", type=int, default=5
    )
    parser.add_argument(
        "--imgs", help="Whether to create images", type=bool, default=False
    )
    args = parser.parse_args()

    if args.strategy not in strategies:
        raise ValueError(f"Invalid strategy: {args.strategy}. Choose from {strategies}.")

    # Load the data
    data = np.load(args.src)

    # Generate the score matrices
    score_matrix = generate_score_matrix(data, r=2)

    if "LAMBERTIAN" in args.strategy:
        leds = get_tx_positions()
        m = - np.log(2) / np.log(np.cos(np.pi / 12))

    h, w, _ = score_matrix.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((yy.ravel(), xx.ravel()), axis=-1)
    for i in tqdm(range(score_matrix.shape[2]), "Cleaning data for LEDs"):
        # Select the data and score matrix for the current LED
        data_i = data[:, :, i]
        score_matrix_i = score_matrix[:, :, i]

        bad = score_matrix_i > score_threshold
        good = ~bad & (data_i != -1) # Good points are those that are not bad and have data

        good_coords = coords[good.ravel()]
        bad_coords = coords[bad.ravel()]

        tree = cKDTree(good_coords)
        dists, idxs = tree.query(bad_coords, k=args.k)

        for j, (y_bad, x_bad) in enumerate(bad_coords):
            d_good = dists[j]
            ixs_good = idxs[j]

            weights = 1 / d_good
            weights /= np.sum(weights)

            if args.strategy == "MEAN":
                # Replace bad point with mean of good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]
                data_i[y_bad, x_bad] = np.mean(rss_values)
           
            elif args.strategy == "IDW":
                # Replace bad point with IDW of good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]
                data_i[y_bad, x_bad] = np.sum(rss_values * weights)

            elif args.strategy == "LAMBERTIAN":
                # Replace bad point using Lambertian falloff
                # Get the RSS values of the good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]

                # Get the RSS value of the first good point (reference)
                rss_ref = rss_values[0]

                # Get the coordinates of the first good point
                y_good, x_good = good_yx[0]

                # Distance from LED to each point
                d1 = np.linalg.norm(leds[i] - np.array([x_bad, y_bad, 0])*10)
                d2 = np.linalg.norm(leds[i] - np.array([x_good, y_good, 0])*10)

                # Replace bad point using Lambertian falloff
                data_i[y_bad, x_bad] = reconstruct_rss_lambertian(rss_ref, d1, d2, m)

            elif args.strategy == "LAMBERTIAN-IDW":
                # Replace bad point using Lambertian falloff
                # Get the RSS values of the good points
                good_yx = good_coords[ixs_good]
                rss_values = data_i[good_yx[:, 0], good_yx[:, 1]]

                # Calculate distances from the LED to the bad point and each good point
                bad_point_3d = np.array([x_bad, y_bad, 0]) * 10 # shape: (3,), 10 is the space between points in mm
                good_points_3d = np.column_stack((good_yx[:, 1], good_yx[:, 0], np.zeros(args.k))) * 10  # shape: (k, 3)

                d1 = np.linalg.norm(leds[i] - bad_point_3d)
                d2 = np.linalg.norm(leds[i] - good_points_3d, axis=1) # shape: (k,)

                # Reconstruct RSS values
                reconstructed_rss = reconstruct_rss_lambertian(rss_values, d1, d2, m)  # shape: (k,)
                data_i[y_bad, x_bad] = np.sum(reconstructed_rss * weights)
    
    print(f"Exporting cleaned data to {args.dst}/cleaned_{args.strategy}.npy")

    os.makedirs(args.dst, exist_ok=True)

    np.save(args.dst + f"/cleaned_{args.strategy}.npy", data)

    if not args.imgs:
        return

    data = np.clip(data, 0, None) # Clip negative values to 0
    for i in tqdm(range(data.shape[2]), "Exporting heat maps for cleaned data"):
        plt.imshow(data[:, :, i], interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.title(f"Cleaned Data for LED {i} ({args.strategy})")
        plt.savefig(args.dst + f"/led_{i}_cleaned_{args.strategy}.png")
        plt.clf()

if __name__ == "__main__":
    main()