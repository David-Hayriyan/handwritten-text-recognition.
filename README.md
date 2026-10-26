# Handwritten Text Recognition

**A PyTorch portfolio project for classifying isolated handwritten digits and Latin letters.**

Built from David Hayriyan's coursework, *“Система распознавания рукописного текста на основе нейронных сетей”* (Neural Network–Based Handwritten Text Recognition System). The recovered CNN is now accompanied by a reproducible training pipeline, an independent validation split, early stopping, evaluation tools, and image inference.

[Open in Colab](https://colab.research.google.com/github/David-Hayriyan/handwritten-text-recognition./blob/main/notebooks/Handwritten_Text_Recognition.ipynb) · [Technical report](docs/REPORT.md) · [Source provenance](docs/PROVENANCE.md)

> **Status:** recovered coursework + refactored implementation. The refactored model has not been trained or benchmarked on EMNIST/MNIST during this preparation. No pretrained weights are included. Historical notebook logs are preserved and clearly distinguished from new results.

## What it does

- Classifies one isolated character per image, with top-k scores.
- Supports EMNIST Balanced (47 classes) and MNIST (10 digits).
- Uses seeded, stratified train/validation splitting, training-only augmentation, Adam, a learning-rate scheduler, and early stopping.
- Saves the best model by validation loss, epoch metrics, split indices, configuration, and environment details.
- Produces test accuracy, macro F1, per-class precision/recall/F1, learning curves, and a confusion matrix.
- Provides an eight-trial validation-only hyperparameter grid and offline pipeline tests.

This is a character classifier. Word/line recognition, Cyrillic, text detection, segmentation, and document OCR are outside its scope.

## Dataset

| Dataset | Classes | Official training set | Validation held out | Training remainder | Official test set |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMNIST Balanced | 47 | 112,800 | 11,280 | 101,520 | 18,800 |
| MNIST | 10 | 60,000 | ≈6,000 | ≈54,000 | 10,000 |

Counts use the default 10% stratified holdout, rounded within each class; MNIST holdout counts are approximate. EMNIST Balanced merges several visually similar uppercase/lowercase classes; it does not provide 62 distinct alphanumeric classes. The mapping is `0–9, A–Z, a b d e f g h n q r t`. See [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) and the [Torchvision dataset implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html).

Datasets are downloaded on demand with `--download` and kept in ignored `data/`. Dataset files and extracted sample images are not shipped. EMNIST's download archive is large and extraction needs additional disk space.

**Preprocessing:** transpose raw EMNIST images to upright orientation; use 28×28 grayscale input; scale to [0,1]; normalize using fixed mean 0.1307 and standard deviation 0.3081 inherited from the coursework. These are MNIST constants, not statistics estimated from EMNIST. Training augmentation applies rotation up to 10° and translation up to 10%. Validation and test receive no random augmentation.

## Architecture

| Stage | Layers | Output per image |
| --- | --- | --- |
| Input | Grayscale | 1 × 28 × 28 |
| Block 1 | Conv 1→32, BN, ReLU; Conv 32→32, BN, ReLU; MaxPool; Dropout2d | 32 × 14 × 14 |
| Block 2 | Conv 32→64, BN, ReLU; Conv 64→64, BN, ReLU; MaxPool; Dropout2d | 64 × 7 × 7 |
| Block 3 | Conv 64→128, BN, ReLU; MaxPool; Dropout2d | 128 × 3 × 3 |
| Classifier | Flatten → Linear 1152→256 → BN → ReLU → Dropout → Linear | 47 or 10 logits |

All convolutions use 3×3 kernels and padding 1. Convolutional dropout is 0.25; classifier dropout defaults to 0.5. The 47-class network has **447,247 trainable parameters**, matching the recovered architecture. Cross entropy uses label smoothing 0.1 for training/validation; final test loss uses ordinary cross entropy.

## Run locally

Use **Python 3.12** and a fresh environment. From a terminal:

```bash
git clone https://github.com/David-Hayriyan/handwritten-text-recognition..git handwritten-text-recognition
cd handwritten-text-recognition
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the CPU build and pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

For NVIDIA GPU environments, select an appropriate build from the [official PyTorch installer](https://pytorch.org/get-started/locally/). CPU execution is supported. GPU results and runtime have not been verified for this refactor.

Train and visualize:

```bash
python -m htr.train --dataset emnist-balanced --download --epochs 30 --patience 5 --output runs/emnist-baseline
python -m htr.plot_history --run runs/emnist-baseline
```

For a smaller digits-only dataset, replace `--dataset emnist-balanced` with `--dataset mnist` and choose a fresh output directory.

Training never reads the official test split. The last incomplete training batch is dropped to avoid singleton BatchNorm batches; shuffling changes the omitted examples each epoch. Existing run folders are rejected to preserve previous work.

After choosing a model using validation only, evaluate its frozen checkpoint:

```bash
python -m htr.evaluate --checkpoint runs/emnist-baseline/best.pt --download --output runs/emnist-test
```

## Predict your own image

After training, save a single upright character on a plain background outside the repository and run:

```bash
python -m htr.predict --checkpoint runs/emnist-baseline/best.pt --image ../my-character.png --top-k 5 --save-preprocessed runs/input-28x28.png
```

The command prints JSON with characters, class indices, and softmax scores. It crops the foreground, preserves aspect ratio, fits the glyph into 20×20, and centers it on a 28×28 canvas. Inspect the saved input to diagnose preprocessing problems. See [the inference guide](examples/README.md).

Auto polarity uses the image border; override it with `--polarity dark-on-light` or `--polarity light-on-dark`. Softmax scores are not calibrated confidence estimates. Performance on phone photos, unusual handwriting, or unseen symbols is unknown.

## Google Colab

Open [the notebook](https://colab.research.google.com/github/David-Hayriyan/handwritten-text-recognition./blob/main/notebooks/Handwritten_Text_Recognition.ipynb) and run the setup and training cells. The supplied setup installs a CPU build. To use a GPU, select a GPU runtime and replace that install command with the matching official CUDA build. Evaluation is a separate explicit step. Upload your own character for inference after a checkpoint exists. Download run artifacts before ending the temporary Colab session.

The notebook is intentionally delivered without executed outputs. Its code imports the same modules as local runs.

## Results and evidence

| Implementation | Recorded accuracy | Interpretation |
| --- | ---: | --- |
| Historical PyTorch EMNIST notebook, May copy | 88.88% | Saved log, selected using the official test split |
| Historical PyTorch EMNIST notebook, June copy | 88.63% | Saved log, selected using the official test split |
| Historical TensorFlow MNIST baseline | 99.26% | Saved test-evaluation output; no recovered checkpoint or independent rerun |
| Refactored PyTorch pipeline | Pending | Requires dataset training and held-out test evaluation |

Historical numbers come from [preserved log excerpts](docs/evidence/). They are **not independently reproduced benchmark results**. In the original EMNIST loop, the official test set was called “validation” and influenced checkpoint selection and the scheduler. Its accuracy is therefore not an untouched final-test estimate. The revised orientation and evaluation protocol also make direct comparisons inappropriate.

No metrics, confusion-matrix image, or learning-curve image from an unperformed new experiment are presented as results. See [the report](docs/REPORT.md) for the evaluation protocol and pending experiments.

## Hyperparameter study

```bash
python -m htr.experiments --dataset emnist-balanced --download --epochs 30 --output runs/grid
```

This runs 2 learning rates × 2 dropout rates × augmentation on/off with the same seed and split. Compare `validation_comparison.csv`, choose a checkpoint using validation loss, then perform final test evaluation once. The grid is implemented; it has not been run on the dataset.

## Project layout

```text
htr/                    # model, data, training, inference, evaluation, experiments
notebooks/
  Handwritten_Text_Recognition.ipynb
  archive/              # cleaned recovered coursework sources
docs/
  REPORT.md             # technical report and evaluation plan
  PROVENANCE.md         # recovered sources, changes, and evidence boundaries
  VALIDATION.md         # checks actually performed during preparation
  evidence/             # historical text logs and source fingerprints
examples/               # user-image inference guide
tests/                  # offline behavior and pipeline checks
requirements*.txt       # pinned runtime and development dependencies
```

Run checks with `python -m pytest -q`. Tests use synthetic data, never download datasets, and do not establish recognition accuracy.

## Author and provenance

**David Hayriyan — [David-Hayriyan](https://github.com/David-Hayriyan)**

The architecture and historical baseline were recovered from the author's existing coursework materials. The modular pipeline, tests, English documentation, and Colab entry point were prepared with Codex assistance in October 2026. See [provenance](docs/PROVENANCE.md) for exactly what was recovered and what was newly implemented.

Dataset and dependency rights remain with their respective owners. No dataset, private user image, secret, credential, or pretrained model is included. No blanket license is assigned to recovered materials in this publication.
