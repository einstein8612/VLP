import argparse
import os

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

leds = [[22.605263157894736, 17.605263157894736, 176.0], [67.5, 19.0, 176.0], [118.86503067484662, 16.098159509202453, 176.0], [168.00602409638554, 18.99397590361446, 176.0], [216.4935064935065, 17.983766233766232, 176.0], [266.5068493150685, 16.56164383561644, 176.0], [21.42697907188353, 71.89672429481347, 176.0], [67.36892003297609, 63.80956306677659, 176.0], [116.05555555555556, 71.0, 176.0], [166.5, 67.5, 176.0], [211.04736842105265, 58.218947368421055, 176.0], [274.5, 64.5, 176.0], [18.03846153846154, 122.03846153846153, 176.0], [68.23500749625187, 121.25824587706147, 176.0], [119.46458087367178, 121.48347107438016, 176.0], [169.91176470588235, 122.0, 176.0], [218.5, 123.5, 176.0], [267.5302521008403, 118.28151260504201, 176.0], [17.875, 175.17499999999998, 176.0], [69.69359658484525, 172.70021344717182, 176.0], [117.50792393026941, 169.0419968304279, 176.0], [171.5, 170.5, 176.0], [216.47969543147207, 171.03807106598984, 176.0], [271.1503957783641, 170.2678100263852, 176.0], [25.554163845633038, 221.02403520649966, 176.0], [56.96808510638298, 227.98936170212767, 176.0], [117.71626583440425, 224.83761703690104, 176.0], [169.46743447180302, 223.91620333598095, 176.0], [212.7557286892759, 224.65930339138407, 176.0], [268.2879069767442, 223.02093023255816, 176.0], [24.210526315789473, 270.7368421052631, 176.0], [67.4811320754717, 268.5188679245283, 176.0], [124.02187182095626, 269.3565615462869, 176.0], [173.34259259259258, 269.3148148148148, 176.0], [224.48444747612552, 269.7387448840382, 176.0], [268.758064516129, 271.9537634408602, 176.0]]

# leds = np.array([[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
#             [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
#             [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
#             [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
#             [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
#             [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]]) / 10.0

def reconstruct_rss_lambertian(rss_ref, d1, d2, m = 19.9937273585):
    r"""
    Reconstructs RSS at d1 using known RSS at d2 with Lambertian model.

    rss_ref: RSS value at d2
    d1: Distance from LED to the point to reconstruct (bad point)
    d2: Distance from LED to the reference point (good point)
    m: Lambertian exponent (calculated from the LED positions)

    Check clean.py for the proof of this formula.
    """

    exponent = m + 3
    return rss_ref * (d2 / d1) ** exponent

def upsample_to_grid(data: npt.NDArray[np.float64], factor: int, fill_value=-1) -> npt.NDArray[np.float64]:
    grid = np.full((data.shape[0]*factor, data.shape[1]*factor, data.shape[2]), fill_value, dtype=data.dtype)
    grid[::factor, ::factor] = data
    return grid

def augment_data(data: npt.NDArray[np.float64], fill_value=-1, q: int = 1, offset_x: int = 0, offset_y: int = 0) -> npt.NDArray[np.float64]:
    y_indices, x_indices, l_indices = np.where(data == fill_value)

    for y, x, no_led in tqdm(zip(y_indices, x_indices, l_indices), total=len(y_indices), desc=f"Augmenting data q={q}/4"):
        amt = 0

        data[y, x, no_led] = 0

        nearest_y = np.clip(int(np.ceil(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.ceil(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1

        nearest_y = np.clip(int(np.floor(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.floor(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1

        nearest_y = np.clip(int(np.ceil(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.floor(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1

        nearest_y = np.clip(int(np.floor(y / 4) * 4), 0, data.shape[0] - 1)
        nearest_x = np.clip(int(np.ceil(x / 4) * 4), 0, data.shape[1] - 1)

        d1 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([x, y, 0]))
        d2 = np.linalg.norm(leds[no_led] + np.array([offset_x, offset_y, 0]) - np.array([nearest_x, nearest_y, 0]))

        reconstructed_rss = reconstruct_rss_lambertian(data[nearest_y, nearest_x, no_led], d1, d2)
        if reconstructed_rss > 0:
            data[y, x, no_led] += reconstructed_rss
            amt += 1
        
        data[y, x, no_led] /= 4

    return data

def main():
    parser = argparse.ArgumentParser("dataset_downsample")
    parser.add_argument(
        "--src", help="Folder to import from", type=str, default="dataset/heatmaps/heatmap_176_downsampled_4"
    )
    parser.add_argument(
        "--dst", help="File to export to", type=str, default="dataset/heatmaps/heatmap_176_augmented_4_downsampled_4/augmented.npy"
    )
    parser.add_argument(
        "--factor", help="Augmenting factor", type=int, default=4
    )
    parser.add_argument(
        "--cross_x_start", help="Cross x start", type=int, default=121
    )
    parser.add_argument(
        "--cross_x_end", help="Cross x end", type=int, default=161
    )
    parser.add_argument(
        "--cross_y_start", help="Cross y start", type=int, default=121
    )
    parser.add_argument(
        "--cross_y_end", help="Cross y end", type=int, default=155
    )
    parser.add_argument(
        "--final_size_x", help="Final size x", type=int, default=282
    )
    parser.add_argument(
        "--final_size_y", help="Final size y", type=int, default=276
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    print(f"Loading data from {args.src}")
    q1 = np.load(f"{args.src}/q1_downsampled.npy")
    q2 = np.load(f"{args.src}/q2_downsampled.npy")
    q3 = np.load(f"{args.src}/q3_downsampled.npy")
    q4 = np.load(f"{args.src}/q4_downsampled.npy")

    upsampled_q1 = upsample_to_grid(q1, args.factor)
    upsampled_q2 = upsample_to_grid(q2, args.factor)
    upsampled_q3 = upsample_to_grid(q3, args.factor)
    upsampled_q4 = upsample_to_grid(q4, args.factor)

    upsampled_q1 = augment_data(upsampled_q1, q=1, offset_x=args.cross_x_end, offset_y=args.cross_y_end)
    upsampled_q2 = augment_data(upsampled_q2, q=2, offset_x=0, offset_y=args.cross_y_end)
    upsampled_q3 = augment_data(upsampled_q3, q=3, offset_x=0, offset_y=0)
    upsampled_q4 = augment_data(upsampled_q4, q=4, offset_x=args.cross_x_end, offset_y=0)

    upsampled = np.full((args.final_size_y, args.final_size_x, len(leds)), -1, dtype=np.float64)
    upsampled[args.cross_y_end:, args.cross_x_end:] = upsampled_q1[:args.final_size_y-args.cross_y_end, :args.final_size_x-args.cross_x_end]
    upsampled[args.cross_y_end:, :args.cross_x_start] = upsampled_q2[:args.final_size_y-args.cross_y_end, :args.cross_x_start]
    upsampled[:args.cross_y_start, :args.cross_x_start] = upsampled_q3[:args.cross_y_start, :args.cross_x_start]
    upsampled[:args.cross_y_start, args.cross_x_end:] = upsampled_q4[:args.cross_y_start, :args.final_size_x-args.cross_x_end]

    np.save(args.dst, upsampled)
    print(f"Exported data to {args.dst}")

if __name__ == "__main__":
    main()