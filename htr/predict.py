"""Predict one isolated upright character from a local image."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .checkpoint import load_checkpoint
from .data import MEAN, STD


def preprocess_image(path, polarity="auto"):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("L")
    array = np.asarray(image)
    if min(array.shape) < 2:
        raise ValueError("Image must be at least 2x2 pixels")
    border = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])
    if polarity == "dark-on-light" or (polarity == "auto" and np.median(border) > 127):
        image = ImageOps.invert(image)
    array = np.asarray(image)
    if int(array.max()) - int(array.min()) < 8:
        raise ValueError("Image is blank or has insufficient contrast")
    # Subtract remaining background; retain antialiased stroke intensities.
    floor = float(np.median(np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])))
    array = np.clip(array.astype(np.float32) - floor, 0, 255)
    mask = array > max(16, float(array.max()) * 0.15)
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("No foreground character found")
    image = Image.fromarray(array.astype(np.uint8)).crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
    scale = 20 / max(image.size)
    image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28-image.width)//2, (28-image.height)//2))
    tensor = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0)
    tensor = ((tensor - MEAN) / STD).unsqueeze(0).unsqueeze(0)
    return tensor, canvas


def predict(checkpoint_path, image_path, top_k=5, polarity="auto", save_preprocessed=None):
    model, checkpoint = load_checkpoint(checkpoint_path)
    labels = checkpoint["labels"]
    if not 1 <= top_k <= len(labels):
        raise ValueError(f"top_k must be between 1 and {len(labels)}")
    tensor, canvas = preprocess_image(image_path, polarity)
    if save_preprocessed:
        Path(save_preprocessed).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(save_preprocessed)
    with torch.inference_mode():
        scores, indices = model(tensor).softmax(1)[0].topk(top_k)
    return {"dataset": checkpoint["config"]["dataset"], "checkpoint_epoch": checkpoint["epoch"],
            "predictions": [{"character": labels[i], "class_index": i, "softmax_score": s}
                            for s, i in zip(scores.tolist(), indices.tolist())]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--polarity", choices=["auto", "dark-on-light", "light-on-dark"], default="auto")
    p.add_argument("--save-preprocessed")
    args = p.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image, args.top_k, args.polarity,
                             args.save_preprocessed), indent=2))


if __name__ == "__main__":
    main()
