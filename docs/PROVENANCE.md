# Source provenance

## Recovered materials

Three relevant notebooks were found in the author's connected Google Drive and read on 2026-10-27:

| Source title | Drive modification date | Contents |
| --- | --- | --- |
| handwriting_recognition.ipynb | 2026-05-31 | PyTorch CNN, EMNIST Balanced, 15-epoch output, inference and export cells |
| handwriting_recognition (1).ipynb | 2026-06-15 | Same EMNIST architecture and another saved training log |
| digit_recognizer_colab_2.ipynb | 2026-05-16 | TensorFlow/Keras CNN, MNIST, 8-epoch output, custom-image inference |

These dates are file metadata, not independently verified experiment execution dates.
Private Drive IDs and links are omitted from the public repository.

The referenced previous conversation contained no usable implementation attachments.
No matching recognition repository was found among the accessible repositories for
David-Hayriyan. No pretrained checkpoint or exported project archive was recovered
in the searches performed; saved notebook output merely mentions such files.

## What was preserved

- The five-convolution PyTorch model was extracted from the May EMNIST notebook.
  Its module names, BatchNorm layers, tensor dimensions, and original initialization
  are retained; class count and classifier dropout are configurable.
- Cleaned May EMNIST and MNIST notebooks are in notebooks/archive/.
- Historical training stdout is preserved in docs/evidence/. Numbers were not
  generated or rerun during this preparation.
- The June EMNIST copy provides a second log, not an additional independent
  verification of the May run.
- source_manifest.json fingerprints the locally materialized, formatted JSON
  source snapshots. Those hashes are not claimed to be hashes of the original
  Drive byte streams or of the cleaned public notebooks.

Archive cleaning removes embedded image outputs, all remaining outputs, execution
counts, and Colab-specific metadata. For EMNIST, export/download cells 11–12 were
omitted. Source code in the remaining cells is preserved, including historical
limitations. Local /content/ paths are retained for context. Archived notebooks
are reference material rather than maintained entry points.

## Newly implemented with Codex assistance

The htr modules, tests, English README/report, and main Colab notebook were created
or refactored for this portfolio publication. They are not represented as code
already present in the coursework.

Changes include train/validation separation, upright EMNIST orientation, independent
deterministic validation transforms, early stopping, versioned checkpoints,
run metadata, JSON inference, robust image preprocessing, per-class metrics,
a confusion matrix, and a validation-only experiment grid.

New checkpoints deliberately require a format/preprocessing version and cannot
silently load old coursework weights. Reusing an unknown legacy orientation could
produce plausible but incorrect predictions.

## Evidence limits

The original EMNIST loop passed test_loader to validate() every epoch, used its
loss for ReduceLROnPlateau, and selected checkpoints by its accuracy. Historical
88.88% and 88.63% therefore reflect test-informed model selection.

The TensorFlow notebook records 0.9926 test accuracy after an eight-epoch fit.
That is a historical output observation, not an independently verified model score.
Neither its weights nor a complete experiment environment were recovered.

No full dataset training, hyperparameter study, or final test evaluation of the
new implementation has been performed during project preparation. Offline tests
are described in VALIDATION.md. Original files remain in the private workspace;
only the cleaned relevant materials are published.

## Rights and attribution

The user supplied the coursework context and authorized publication under
David-Hayriyan. This repository does not establish third-party ownership or assign
a new license to any pre-existing third-party material. Dataset files and embedded
sample/user images are excluded. PyTorch, Torchvision, NumPy, Pillow, Matplotlib,
and scikit-learn retain their own licenses. Consult the linked dataset providers
before redistributing data.
