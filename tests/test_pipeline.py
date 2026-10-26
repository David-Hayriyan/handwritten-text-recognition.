"""Offline tests: no datasets downloaded and no benchmark accuracy asserted."""
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, TensorDataset

from htr.checkpoint import load_checkpoint
from htr.data import BALANCED_LABELS, DatasetView, correct_orientation, make_transform, stratified_split
from htr.evaluate import write_evaluation
from htr.model import HandwritingCNN
from htr.predict import predict, preprocess_image
from htr.train import fit, parser


@pytest.fixture(autouse=True)
def fast_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def test_stratified_split_is_disjoint_complete_and_repeatable():
    targets = np.repeat(np.arange(47), 20)
    train, val = stratified_split(targets)
    assert len(val) == 94
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(940))
    assert all(sum(targets[i] == label for i in val) == 2 for label in range(47))
    assert (train, val) == stratified_split(targets)
    assert (train, val) != stratified_split(targets, seed=7)


def test_labels_match_emnist_balanced():
    assert len(BALANCED_LABELS) == len(set(BALANCED_LABELS)) == 47
    assert BALANCED_LABELS[36:] == list("abdefghnqrt")


def test_emnist_orientation_and_eval_transform():
    array = np.zeros((28, 28), dtype=np.uint8)
    array[3:8, 10:12] = 255
    upright = Image.fromarray(array)
    raw = correct_orientation(upright)
    expected = make_transform("mnist")(upright)
    actual = make_transform("emnist-balanced")(raw)
    assert torch.equal(actual, expected)
    assert torch.equal(actual, make_transform("emnist-balanced")(raw))


def test_validation_view_does_not_share_training_augmentation():
    image = Image.new("L", (28, 28))
    training = DatasetView([(image, 1)], [0], lambda x: torch.ones(1,28,28))
    validation = DatasetView(training.dataset, [0], make_transform("mnist"))
    assert not torch.equal(training[0][0], validation[0][0])
    assert torch.equal(validation[0][0], validation[0][0])


@pytest.fixture
def character(tmp_path):
    image = Image.new("L", (80, 100), 255)
    ImageDraw.Draw(image).line([(20, 20), (60, 20), (20, 80)], fill=0, width=8)
    path = tmp_path / "character.png"
    image.save(path)
    return path


def test_preprocess_polarity_and_blank_rejection(character, tmp_path):
    tensor, canvas = preprocess_image(character)
    assert tensor.shape == (1, 1, 28, 28)
    assert torch.isfinite(tensor).all()
    assert canvas.getbbox() is not None
    inverted = tmp_path / "inverted.png"
    ImageOps.invert(Image.open(character)).save(inverted)
    assert torch.equal(tensor, preprocess_image(inverted)[0])
    blank = tmp_path / "blank.png"
    Image.new("L", (28,28), 255).save(blank)
    with pytest.raises(ValueError, match="blank"):
        preprocess_image(blank)


def test_forward_backward_checkpoint_predict_and_evaluation(tmp_path, character):
    config = vars(parser().parse_args(["--dataset", "mnist", "--epochs", "1", "--batch-size", "4"]))
    data = TensorDataset(torch.randn(8,1,28,28), torch.arange(8) % 10)
    loader = DataLoader(data, batch_size=4)
    model = HandwritingCNN(10)
    initial = model.fc2.weight.detach().clone()
    run = tmp_path / "run"
    summary = fit(model, loader, loader, config, run, torch.device("cpu"), {"synthetic": True})
    assert summary["epochs_completed"] == 1 and not summary["test_evaluated"]
    assert not torch.equal(initial, model.fc2.weight)
    restored, checkpoint = load_checkpoint(run / "best.pt")
    assert restored(torch.zeros(1,1,28,28)).shape == (1,10)
    assert checkpoint["epoch"] == 1
    result = predict(run / "best.pt", character, top_k=1)
    assert len(result["predictions"]) == 1
    assert 0 <= result["predictions"][0]["softmax_score"] <= 1
    with pytest.raises(ValueError, match="top_k"):
        predict(run / "best.pt", character, top_k=11)
    metrics = write_evaluation([0,1,1], [0,0,1], list("01"),
                               {"accuracy": 2/3}, tmp_path / "eval")
    assert metrics["macro_f1"] == pytest.approx(2/3)
    matrix = np.loadtxt(tmp_path / "eval/confusion_matrix.csv", delimiter=",")
    assert matrix.sum() == 3
    assert (tmp_path / "eval/confusion_matrix.png").is_file()
    assert json.loads((run / "training_summary.json").read_text())["test_evaluated"] is False


def test_early_stopping_saves_best_validation_epoch(tmp_path):
    config = vars(parser().parse_args(["--dataset","mnist","--epochs","10","--patience","2"]))
    values = [1.0, 0.9, 0.95, 0.96]
    calls = []
    def fake_epoch(*args, **kwargs):
        training = len(args) > 3 and args[3] is not None
        if training:
            return {"loss": 1.0, "accuracy": 0.1, "samples": 8}, [], []
        loss = values[len(calls)]
        calls.append(loss)
        return {"loss": loss, "accuracy": 0.2, "samples": 8}, [], []
    with patch("htr.train.run_epoch", side_effect=fake_epoch):
        summary = fit(HandwritingCNN(10), [], [], config, tmp_path/"early",
                      torch.device("cpu"), {"synthetic": True})
    assert summary["best_epoch"] == 2
    assert summary["epochs_completed"] == 4
    _, checkpoint = load_checkpoint(tmp_path/"early/best.pt")
    assert checkpoint["epoch"] == 2


def test_legacy_checkpoint_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_checkpoint(path)


def test_public_cli_roundtrip_on_synthetic_idx(tmp_path, character):
    import os
    import struct
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / 'data'
    raw = data_root / 'MNIST' / 'raw'
    raw.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for prefix, count in [('train', 40), ('t10k', 10)]:
        images = rng.integers(0, 256, size=(count,28,28), dtype=np.uint8)
        labels = np.arange(count, dtype=np.uint8) % 10
        (raw / (prefix+'-images-idx3-ubyte')).write_bytes(struct.pack('>IIII',2051,count,28,28)+images.tobytes())
        (raw / (prefix+'-labels-idx1-ubyte')).write_bytes(struct.pack('>II',2049,count)+labels.tobytes())
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    def run(module, *args):
        return subprocess.run([sys.executable,'-m',module,*map(str,args)], cwd=root,
                              env=env, capture_output=True, text=True, check=True)
    run_dir = tmp_path / 'cli-run'
    run('htr.train','--dataset','mnist','--data-dir',data_root,'--epochs','1',
        '--batch-size','4','--output',run_dir,'--device','cpu')
    run('htr.plot_history','--run',run_dir)
    evaluation = tmp_path / 'cli-test'
    run('htr.evaluate','--checkpoint',run_dir/'best.pt','--data-dir',data_root,
        '--output',evaluation)
    result = run('htr.predict','--checkpoint',run_dir/'best.pt','--image',character,'--top-k','1')
    assert len(json.loads(result.stdout)['predictions']) == 1
    assert json.loads((evaluation/'metrics.json').read_text())['samples'] == 10
    assert (run_dir/'learning_curves.png').is_file()


def test_notebooks_are_valid_and_main_has_no_claimed_outputs():
    import nbformat
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for path in (root/'notebooks').rglob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert all(not cell.get('outputs') for cell in notebook.cells)


def test_parameter_count_matches_coursework():
    assert sum(p.numel() for p in HandwritingCNN(47).parameters()) == 447247
