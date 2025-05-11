import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import r3fit
from tqdm import tqdm


def kasa_circle_fit(points: np.ndarray):
    """
    Chernov-style Kåsa circle fit using moment-based algebraic method.

    Basically taken directly from:
    https://people.cas.uab.edu/~mosya/cl/CircleFitByKasa.cpp
    after being translated to Python.
    
    Parameters:
        points: np.ndarray of shape (N, 2), each row is (x, y)
    
    Returns:
        (a, b, r, sigma) -> center (a, b), radius r, RMS error sigma
    """
    if points.shape[1] != 2:
        raise ValueError("Input must be an Nx2 array of (x, y) points.")
    N = points.shape[0]
    if N < 3:
        raise ValueError("At least 3 points are required to fit a circle.")

    X = points[:, 0]
    Y = points[:, 1]

    mean_x = X.mean()
    mean_y = Y.mean()

    # Center the coordinates
    Xi = X - mean_x
    Yi = Y - mean_y
    Zi = Xi**2 + Yi**2

    # Compute moments
    Mxx = np.mean(Xi**2)
    Myy = np.mean(Yi**2)
    Mxy = np.mean(Xi * Yi)
    Mxz = np.mean(Xi * Zi)
    Myz = np.mean(Yi * Zi)

    # Cholesky-like decomposition and solving
    G11 = np.sqrt(Mxx)
    G12 = Mxy / G11
    G22 = np.sqrt(Myy - G12**2)

    D1 = Mxz / G11
    D2 = (Myz - D1 * G12) / G22

    C = D2 / (2 * G22)
    B = (D1 - G12 * C) / (2 * G11)

    a = B + mean_x
    b = C + mean_y
    r = np.sqrt(B**2 + C**2 + Mxx + Myy)

    # Compute RMS error (sigma)
    distances = np.sqrt((X - a)**2 + (Y - b)**2)
    sigma = np.sqrt(np.mean((distances - r)**2))

    return a, b, r, sigma


def get_tx_positions():
    ##Actual position of the TX (roughly measured)
    return [[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
            [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
            [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
            [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
            [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
            [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]]

def main():
    parser = argparse.ArgumentParser("find_led_centers")
    parser.add_argument(
        "--src", help="File to import from", type=str, default="dataset/heatmaps/heatmap_176/raw.npy"
    )
    parser.add_argument(
        "--min_sample", help="Minimum sample forming circle", type=float, default=0.395
    )
    parser.add_argument(
        "--max_sample", help="Maximum sample forming circle", type=float, default=0.405
    )
    parser.add_argument(
        "--imgs", help="Whether or not to generate images indicating differences", type=bool, default=False
    )
    
    args = parser.parse_args()

    if args.imgs:
        os.makedirs("leds", exist_ok=True)

    data = np.load(args.src)
    data = data.mean(axis=3).clip(0, 1)

    circle_mask = (data > args.min_sample) & (data < args.max_sample)
    circle_points = np.argwhere(circle_mask)

    led_positions = []

    for i in tqdm(range(data.shape[2]), desc="Fitting circles"):
        relevant_points = circle_points[circle_points[:, 2] == i][:, [1,0]].astype(np.float64)
        circle = r3fit.fit(relevant_points, 10000, 0.05)
        kasa_circle = kasa_circle_fit(relevant_points)

        led_positions.append((circle.x, circle.y))

        if not args.imgs:
            continue

        plt.imshow(data[:, :, i], origin="lower")

        # Plot the points that are used to fit the circle
        plt.scatter(relevant_points[:, 0], relevant_points[:, 1], c="purple", s=1, label="Points used to fit circle")

        # Plot the fitted circle
        circle_x = circle.x + circle.r * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_y = circle.y + circle.r * np.sin(np.linspace(0, 2 * np.pi, 100))
        plt.plot(circle_x, circle_y, color="red", linewidth=2)

        # Plot the Kåsa circle
        kasa_circle_x = kasa_circle[0] + kasa_circle[2] * np.cos(np.linspace(0, 2 * np.pi, 100))
        kasa_circle_y = kasa_circle[1] + kasa_circle[2] * np.sin(np.linspace(0, 2 * np.pi, 100))
        plt.plot(kasa_circle_x, kasa_circle_y, color="green", linewidth=2)

        # Plot the TX positions
        plt.scatter(circle.x, circle.y, c="red", s=1, label="R3Fit (Us)")
        plt.scatter(kasa_circle[0], kasa_circle[1], c="green", s=1, label="Kasa (Old)")
        plt.scatter(get_tx_positions()[i][0]/10, get_tx_positions()[i][1]/10, c="blue", s=1, label="Original TX Position")

        plt.colorbar()
        plt.title(f"LED {i} - Circle Fit Comparison")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        plt.xlim(0, data.shape[1])
        plt.ylim(0, data.shape[0])

        plt.savefig(f"leds/led_{i}_circle_fit.png")
        
        plt.clf()
    
    # Save the LED positions to a file
    json.dump(led_positions, open("leds/led_positions.json", "w"))

    if not args.imgs:
        return
    # Show all the LED positions on a single plot
    plt.imshow(np.zeros_like(data[:, :, 0]), origin="lower")
    plt.scatter([pos[0] for pos in led_positions], [pos[1] for pos in led_positions], c="red", s=1)
    plt.scatter([pos[0]/10 for pos in get_tx_positions()], [pos[1]/10 for pos in get_tx_positions()], c="blue", s=1)
    plt.title("LED Positions")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xlim(0, data.shape[1])
    plt.ylim(0, data.shape[0])
    plt.savefig("leds/led_positions.png")

if __name__ == "__main__":
    main()