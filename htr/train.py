"""Train on training/validation data; test evaluation is a separate command."""
import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from .checkpoint import FORMAT_VERSION, PREPROCESSING
from .data import DatasetView, labels_for, load_dataset, make_transform, set_seed, stratified_split
from .engine import run_epoch
from .model import HandwritingCNN


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["emnist-balanced", "mnist"], default="emnist-balanced")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="runs/emnist-baseline")
    p.add_argument("--download", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def fit(model, train_loader, val_loader, config, output, device, split_indices):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "split_indices.json").write_text(json.dumps(split_indices), encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "torchvision": str(torchvision.__version__), "device": str(device),
                   "git_revision": revision, "seed": config["seed"]}
    if str(device).startswith("cuda"):
        environment["gpu"] = torch.cuda.get_device_name(device)
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best_loss, stale, best_epoch = float("inf"), 0, 0
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr", "seconds"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train, _, _ = run_epoch(model, train_loader, device, optimizer, config["label_smoothing"])
            val, _, _ = run_epoch(model, val_loader, device, label_smoothing=config["label_smoothing"])
            row = dict(epoch=epoch, train_loss=train["loss"], train_accuracy=train["accuracy"],
                       val_loss=val["loss"], val_accuracy=val["accuracy"], lr=lr,
                       seconds=time.perf_counter() - start)
            writer.writerow(row)
            f.flush()
            print(json.dumps(row), flush=True)
            scheduler.step(val["loss"])
            if val["loss"] < best_loss:
                best_loss, stale, best_epoch = val["loss"], 0, epoch
                torch.save({"format_version": FORMAT_VERSION, "preprocessing": PREPROCESSING,
                            "model_state": model.state_dict(), "config": config,
                            "labels": labels_for(config["dataset"]), "epoch": epoch,
                            "val_metrics": val}, output / "best.pt")
            else:
                stale += 1
                if stale >= config["patience"]:
                    break
    summary = {"best_epoch": best_epoch, "best_val_loss": best_loss,
               "epochs_completed": epoch, "test_evaluated": False}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1 or args.lr <= 0:
        raise SystemExit("epochs/patience/lr must be positive; batch-size must be >= 2")
    if not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1 or args.weight_decay < 0:
        raise SystemExit("Invalid dropout, label smoothing, or weight decay")
    if Path(args.output).exists():
        raise SystemExit("Output exists; choose a fresh --output to preserve previous runs")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    raw = load_dataset(args.dataset, args.data_dir, train=True, download=args.download)
    train_indices, val_indices = stratified_split(raw.targets, args.val_fraction, args.seed)
    training = DatasetView(raw, train_indices, make_transform(args.dataset, not args.no_augment))
    validation = DatasetView(raw, val_indices, make_transform(args.dataset))
    # BatchNorm1d needs at least two training examples per batch.
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = HandwritingCNN(len(labels_for(args.dataset)), args.dropout).to(device)
    fit(model, train_loader, val_loader, vars(args), args.output, device,
        {"train": train_indices, "validation": val_indices})


if __name__ == "__main__":
    main()
