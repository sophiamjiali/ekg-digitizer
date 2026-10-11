# Model Selection (#13)

## Selected model

A multi-stage heatmap pipeline for digitizing EKG images, used with its pretrained weights:

1. Correct the image's rotation (EfficientViT-B2)
2. Detect grid keypoints and warp the image to a standard layout (ConvNeXt UNet, 5 folds)
3. Predict each lead's waveform (ensemble of 2 dual-encoder UNets)
4. Re-check leads with large amplitudes (ConvNeXt UNet)

## Why

- Highest score of the pretrained options available: **22.63 dB SNR** on the competition's hidden test set. The next best scored 22.25 dB.
- Weights, configs and inference code are all available, so we don't need to train anything.
- We use it unchanged, so the score above still applies. Fallback if it's too slow: the single-waveform-model version (22.35 dB).

Not chosen: the other public solutions scored lower or have no weights. Training our own was out of scope.

## Test evaluation

Run once on a held-out test set and don't tune afterwards. Don't use the competition train set, because the model was trained on it.

| Test set | SNR (dB) | MAE (mV) | Date |
|---|---|---|---|
| TODO (#14 data) | | | |

## Registration

TODO once the registry (#12) is set up. Register as `production` with:
- the checkpoints and config `.yaml` for each stage, from upstream commit `8358d5b`
- metrics: 22.63 dB competition score and the test result above

## Sources

- Code: https://github.com/dangnh0611/kaggle_ecg_digitization
- Weights: https://www.kaggle.com/datasets/dangnh0611/ecg-checkpoints
