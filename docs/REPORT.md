# Technical report

## Problem and scope

Classify an isolated handwritten digit or Latin character from a grayscale image.
The project develops the original coursework into an inspectable ML workflow.
It does not transcribe words, recognize Cyrillic, segment pages, or provide a
production OCR service.

## Data and preparation

The default dataset is EMNIST Balanced: 47 classes, 112,800 official training
examples, and 18,800 official test examples. The alternative is MNIST digits.
See [NIST's dataset page](https://www.nist.gov/itl/products-and-services/emnist-dataset)
and [Torchvision's loader](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html).

A seeded stratified 10% holdout is made within the official training set.
The raw storage is shared between two dataset views with different transforms.
Only training gets random affine augmentation. Split indices are saved so the
same examples can be used across experiments. The split is by image, not by
writer; writer-level independence is not established by this implementation.

Raw EMNIST samples are transposed into upright orientation. MNIST is already
upright. Both use fixed MNIST normalization constants 0.1307/0.3081. The retained
constants are a baseline design choice, not measured EMNIST statistics.

## Model and optimization

The recovered network has five 3×3 convolution layers across 32-, 64-, and
128-channel stages, BatchNorm, ReLU, max pooling, and dropout. A 1152→256→47
classifier completes the EMNIST model (447,247 trainable parameters).

Training defaults: Adam, learning rate 0.001, weight decay 0.0001, batch size 128,
up to 30 epochs, label smoothing 0.1, gradient clipping at norm 1.0.
ReduceLROnPlateau monitors validation loss. A checkpoint is saved on strictly
lower validation loss; five epochs without improvement stop training.

The training loader drops the final incomplete batch because the classifier's
BatchNorm layer requires more than one example in training mode. Validation and
test use every example. Exact numerical equality across hardware/library versions
is not guaranteed by fixed random seeds.

## Evaluation protocol

1. Use only training data for gradients.
2. Use the independent validation subset for scheduler, early stopping, and
   hyperparameter/model selection.
3. Freeze the selected checkpoint.
4. Evaluate once on the official test split.
5. Report test accuracy, macro F1, per-class precision/recall/F1/support, and a
   count-based confusion matrix. Keep class-index mapping next to matrix files.
6. Retain configuration, seed, software versions, checkpoint hash, split indices,
   and epoch history with the experiment.

Training/validation loss includes configured label smoothing. Test loss is plain
cross entropy and is labeled explicitly; those loss values are not directly
interchangeable. Accuracy is stored on a 0–1 scale.

## Historical observations

Saved logs show May EMNIST accuracy 88.88%, June EMNIST accuracy 88.63%, and an
earlier TensorFlow MNIST test output of 99.26%. The text excerpts are in
[evidence/](evidence/); the code is in [the archive](../notebooks/archive/).

The EMNIST scores were selected on the official test split. The original pipeline
also lacked upright orientation correction, so historical inference on upright
user images could disagree with dataset preprocessing. These are historical
observations and methodological limitations, not current benchmark claims.
No recovered checkpoint permits independent verification of those logs.

## New experiment results

**Pending.** Full dataset training and final evaluation of the refactored pipeline
have not been run. No new accuracy, F1, or confusion matrix is claimed.

Offline tests verify behavior on synthetic inputs and a minimal gradient step.
They cannot establish handwriting-recognition quality. See [validation](VALIDATION.md).

## Hyperparameter investigation

The implemented eight-trial grid varies learning rate (0.001/0.0003), classifier
dropout (0.3/0.5), and augmentation (on/off), holding the seed and split fixed.
Select by validation loss; do not repeatedly evaluate test performance per trial.
For a fuller study, repeat the chosen configuration across multiple seeds and
report mean and dispersion before assessing stability.

No ablation or grid-search result is reported until the experiments are executed.

## User-image inference and limitations

Inference handles EXIF orientation and transparency, estimates background polarity,
crops the foreground, resizes while preserving aspect ratio, and centers into a
28×28 canvas. This preprocessing differs from curated benchmark preparation.
Photographs, shadows, noise, unusual stroke widths, and touching characters can
cause errors. Inspect the preprocessed image and use a polarity override if needed.

Softmax scores are not calibrated probabilities of correctness. A closed-set
classifier always selects a known class even for an unseen symbol; blank-image
rejection is only a preprocessing check, not general out-of-distribution detection.

## Next measurable steps

- Execute a baseline EMNIST training run and retain its artifacts.
- Run the validation-only grid; freeze and evaluate the selected checkpoint.
- Inspect the largest off-diagonal confusion pairs without inventing their causes.
- Build a consented, separate user-image evaluation set for domain-shift analysis.
- Consider writer-aware evaluation if writer identifiers become available.
