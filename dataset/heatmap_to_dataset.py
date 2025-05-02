import argparse
import os

from util import heatmap_to_data_split_from_file

def main():
    parser = argparse.ArgumentParser("dataset_heatmap_to_dataset")
    parser.add_argument(
        "--src", help="File to import from", type=str, default="dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy"
    )
    parser.add_argument(
        "--dst", help="Folder to export to", type=str, default="dataset/exported/data_176_cleaned"
    )
    parser.add_argument(
        "--training_fraction", help="Fraction of training set", type=float, default=0.8
    )
    parser.add_argument(
        "--seed", help="Random seed", type=int, default=42
    )
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    train, test = heatmap_to_data_split_from_file(
        args.src,
        args.training_fraction,
        args.seed
    )

    print(f"Exporting data to {args.dst}")
    train.to_csv(os.path.join(args.dst, "train.csv"), index=False)
    test.to_csv(os.path.join(args.dst, "test.csv"), index=False)
    print(f"Exported data to {args.dst}")

if __name__ == "__main__":
    main()