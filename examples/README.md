# Try your own character

Use one upright isolated character on a plain background. Crop out neighboring text,
page borders, and shadows. Save it outside this repository, for example as
`../my-character.png`.

After training:

```bash
python -m htr.predict --checkpoint runs/emnist-baseline/best.pt --image ../my-character.png --top-k 5 --save-preprocessed runs/my-character-28x28.png
```

Inspect the saved 28x28 image if the prediction is poor. Use
`--polarity dark-on-light` for dark ink on white paper or
`--polarity light-on-dark` for white ink on black.

The command prints JSON with class indices, characters, and softmax scores.
No sample prediction is shown here because the refactored model has not been
trained on EMNIST. Softmax scores are not calibrated confidence estimates.
