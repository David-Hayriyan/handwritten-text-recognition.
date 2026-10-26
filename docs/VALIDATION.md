# Validation record

Prepared on 2026-10-27.

## Checks actually performed

- Local Windows CPU environment: Python 3.12.14, PyTorch 2.14.1+cpu, Torchvision 0.29.1+cpu, NumPy 2.5.2, Pillow 12.3.0, Matplotlib 3.11.2, scikit-learn 1.9.1, pytest 9.1.1.
- Command: `python -m pytest -q`. Result: **11 passed in 31.70 seconds**.
- Seeded stratified splits: disjointness, full coverage, per-class balance, and repeatability.
- Explicit EMNIST mapping, upright transpose, and deterministic validation transforms.
- Custom-image polarity handling, 28x28 output, and blank-input rejection.
- Real gradient updates on synthetic tensors, best-checkpoint save/load, inference with top-k=1, and invalid top-k rejection.
- Controlled early-stopping scenario: best epoch retained and training stopped after the configured patience.
- Confusion-matrix counts, macro F1, and PNG output on synthetic class labels.
- Full public command sequence on temporary **synthetic IDX-format MNIST files**: train, plot history, evaluate, predict. No dataset download was used.
- Three notebooks validated against nbformat; all delivered notebooks have no executed outputs.
- Model parameter count checked against the recovered coursework architecture.
- All Python sources parsed successfully. A targeted scan found no credential patterns, private Drive URLs, or user-specific absolute filesystem paths in publishable text. This is not a formal security audit.

The GitHub Actions workflow is supplied to repeat offline checks on Linux. No CI success is claimed by this local record; inspect the repository Actions page for its actual status. The Colab notebook was structurally validated, but was not executed in a Colab session.

## Not performed

No full EMNIST/MNIST training, new benchmark test evaluation, hyperparameter grid, GPU validation, or real user-image accuracy study was performed. Synthetic checks establish execution behavior only. Historical coursework metrics were read from saved output; they were not independently reproduced.
