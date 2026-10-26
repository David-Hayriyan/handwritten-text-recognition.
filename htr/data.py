"""Dataset loading, class mapping, and reproducible train/validation separation."""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

BALANCED_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
MEAN, STD = 0.1307, 0.3081  # Fixed MNIST constants retained from coursework.


def labels_for(dataset):
    if dataset == "emnist-balanced":
        return BALANCED_LABELS.copy()
    if dataset == "mnist":
        return list("0123456789")
    raise ValueError(f"Unsupported dataset: {dataset}")


def correct_orientation(image):
    # Transpose raw EMNIST IDX images into upright orientation.
    return image.transpose(Image.Transpose.TRANSPOSE)


def make_transform(dataset, augment=False):
    steps = [correct_orientation] if dataset == "emnist-balanced" else []
    if augment:
        steps.append(transforms.RandomAffine(10, translate=(0.1, 0.1), fill=0))
    return transforms.Compose(steps + [
        transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))
    ])


def load_dataset(name, root="data", train=True, download=False):
    if name == "emnist-balanced":
        return datasets.EMNIST(root, split="balanced", train=train, download=download)
    if name == "mnist":
        return datasets.MNIST(root, train=train, download=download)
    raise ValueError(f"Unsupported dataset: {name}")


def stratified_split(targets, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        if len(indices) < 2:
            raise ValueError("Each class needs at least two examples")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * val_fraction)))
        val.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class DatasetView(Dataset):
    """Share raw data without sharing random augmentation."""
    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), label


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
