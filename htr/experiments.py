"""Run a small validation-only grid; never evaluate the test split."""
import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["emnist-balanced", "mnist"], default="emnist-balanced")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="runs/grid")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    with (root / "validation_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "lr", "dropout", "augment", "best_epoch", "best_val_loss"])
        writer.writeheader()
        for i, (lr, dropout, augment) in enumerate(product([0.001, 0.0003], [0.3, 0.5], [True, False])):
            run = root / f"trial-{i:02d}"
            command = [sys.executable, "-m", "htr.train", "--dataset", args.dataset,
                       "--data-dir", args.data_dir, "--output", str(run), "--epochs", str(args.epochs),
                       "--seed", str(args.seed), "--lr", str(lr), "--dropout", str(dropout)]
            if args.download:
                command.append("--download")
            if not augment:
                command.append("--no-augment")
            subprocess.run(command, check=True)
            summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
            writer.writerow(dict(run=str(run), lr=lr, dropout=dropout, augment=augment,
                                 best_epoch=summary["best_epoch"], best_val_loss=summary["best_val_loss"]))
            f.flush()


if __name__ == "__main__":
    main()
