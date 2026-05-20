"""
Compare training loss curves across different models.

Usage:
    python plot_loss_comparison.py --dir <folder_with_csvs> [--output loss_comparison.png]

The folder should contain CSV files named after each model (e.g., "ModelA.csv").
Each CSV must have an "avg_train_loss" column. The filename (without .csv) is used as the legend label.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves for multiple models")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing CSV files (one per model)")
    parser.add_argument("--output", type=str, default="loss_comparison.png", help="Output image path")
    args = parser.parse_args()

    csv_files = sorted([f for f in os.listdir(args.dir) if f.endswith(".csv")])
    if not csv_files:
        print(f"No CSV files found in {args.dir}")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:
        model_name = os.path.splitext(csv_file)[0]
        df = pd.read_csv(os.path.join(args.dir, csv_file))
        epochs = range(1, len(df) + 1)
        plt.plot(epochs, df["avg_train_loss"], label=model_name, marker="o", markersize=3)

    plt.xlabel("Epoch")
    plt.ylabel("Avg Train Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
