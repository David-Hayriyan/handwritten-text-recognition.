"""Shared train/evaluation loop with sample-weighted metrics."""
import torch
from torch import nn


def run_epoch(model, loader, device, optimizer=None, label_smoothing=0.0):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, correct, count = 0.0, 0, 0
    truth, predictions = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            predicted = logits.argmax(1)
            count += len(labels)
            total_loss += loss.item() * len(labels)
            correct += (predicted == labels).sum().item()
            if not training:
                truth.extend(labels.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())
    if not count:
        raise ValueError("Empty data loader; use more examples or a smaller batch size")
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}, truth, predictions
