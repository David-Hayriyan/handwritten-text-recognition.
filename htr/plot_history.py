"""Render learning curves from an actual run's history.csv."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = Path(args.run)
    with (run / "history.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        for split in ["train", "val"]:
            ax.plot(epochs, [float(row[f"{split}_{metric}"]) for row in rows], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "learning_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
