"""Evaluate a frozen checkpoint once on the official test split."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import DatasetView, load_dataset, make_transform
from .engine import run_epoch


def write_evaluation(truth, predictions, labels, metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ids = list(range(len(labels)))
    matrix = confusion_matrix(truth, predictions, labels=ids)
    metrics = {**metrics, "macro_f1": float(f1_score(truth, predictions, labels=ids, average="macro", zero_division=0))}
    report = classification_report(truth, predictions, labels=ids, target_names=labels,
                                   output_dict=True, zero_division=0)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    np.savetxt(output / "predictions.csv", np.column_stack([truth, predictions]), fmt="%d",
               delimiter=",", header="true_class,predicted_class", comments="")
    (output / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 11))
    plot = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=ids, yticks=ids, xticklabels=labels, yticklabels=labels,
           xlabel="Predicted character", ylabel="True character", title="Test confusion matrix (counts)")
    ax.tick_params(axis="both", labelsize=7)
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    if Path(args.output).exists():
        raise SystemExit("Output exists; use a fresh output directory")
    model, checkpoint = load_checkpoint(args.checkpoint)
    dataset = checkpoint["config"]["dataset"]
    raw = load_dataset(dataset, args.data_dir, train=False, download=args.download)
    view = DatasetView(raw, list(range(len(raw))), make_transform(dataset))
    loader = DataLoader(view, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics, truth, predictions = run_epoch(model, loader, torch.device("cpu"))
    metrics.update(dataset=dataset, split="official_test", checkpoint_epoch=checkpoint["epoch"],
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   loss_definition="cross_entropy_without_label_smoothing")
    metrics = write_evaluation(truth, predictions, checkpoint["labels"], metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
