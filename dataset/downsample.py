import argparse
import os

import numpy as np

def format_array(arr, values_per_line=12):
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), values_per_line):
        chunk = flat[i:i + values_per_line]
        line = ", ".join(f"{x}" for x in chunk)
        lines.append(line)
    return ",\n    ".join(lines)

def main():
    parser = argparse.ArgumentParser("dataset_downsample")
    parser.add_argument(
        "--src", help="File to import from", type=str, default="dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/heatmaps/heatmap_176_downsampled_4"
    )
    parser.add_argument(
        "--factor", help="Downsampling factor", type=int, default=4
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
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    print(f"Loading data from {args.src}")
    src = np.load(args.src)

    # q1 is the first quandrant of the heatmap, which is the first 121 rows and columns
    q1 = src[args.cross_y_end:, args.cross_x_end:]
    q2 = src[args.cross_y_end:, :args.cross_x_start]
    q3 = src[:args.cross_y_start, :args.cross_x_start]
    q4 = src[:args.cross_y_start, args.cross_x_end:]

    print(f"Downsampling data by a factor of {args.factor}")
    q1_downsampled = q1[::args.factor, ::args.factor]
    q2_downsampled = q2[::args.factor, ::args.factor]
    q3_downsampled = q3[::args.factor, ::args.factor]
    q4_downsampled = q4[::args.factor, ::args.factor]

    print(f"Exporting data to {args.dst}")

    print(f"Saving quandrants as numpy arrays at {args.dst}/q*_downsampled.npy")
    np.save(os.path.join(args.dst, "q1_downsampled.npy"), q1_downsampled)
    np.save(os.path.join(args.dst, "q2_downsampled.npy"), q2_downsampled)
    np.save(os.path.join(args.dst, "q3_downsampled.npy"), q3_downsampled)
    np.save(os.path.join(args.dst, "q4_downsampled.npy"), q4_downsampled)

    c_file = os.path.join(args.dst, "downsampled_data.c")

    print(f"Exporting data to {c_file}")
    with open(c_file, "w") as f:
        f.write(f"const float downsampled_data_q1[] = {{\n    {format_array(q1_downsampled)}\n}};\n")
        f.write(f"const float downsampled_data_q2[] = {{\n    {format_array(q2_downsampled)}\n}};\n")
        f.write(f"const float downsampled_data_q3[] = {{\n    {format_array(q3_downsampled)}\n}};\n")
        f.write(f"const float downsampled_data_q4[] = {{\n    {format_array(q4_downsampled)}\n}};\n")
        f.write(f"unsigned int downsampled_data_q_height = {q1_downsampled.shape[0]};\n")
        f.write(f"unsigned int downsampled_data_q_width = {q1_downsampled.shape[1]};\n")
        f.write(f"unsigned int downsampled_data_q_no_led = {q1_downsampled.shape[2]};\n")
        f.write(f"unsigned int downsampled_data_q_len = {q1_downsampled.size};\n")
        f.write(f"unsigned int downsampled_data_factor = {args.factor};\n\n")
        f.write(f"unsigned int cross_x_start = {args.cross_x_start};\n")
        f.write(f"unsigned int cross_x_end = {args.cross_x_end};\n")
        f.write(f"unsigned int cross_y_start = {args.cross_y_start};\n")
        f.write(f"unsigned int cross_y_end = {args.cross_y_end};\n")


    print(f"Exported data to {args.dst}")

if __name__ == "__main__":
    main()