# Model Selection

Issue: #13 Select and Register Best Model

## Decision

We use the **5th place solution** from the Kaggle [PhysioNet - Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization) competition, by dangnh0611. We take its **final submission configuration unchanged** and do not train our own model.

- Code: [dangnh0611/kaggle_ecg_digitization](https://github.com/dangnh0611/kaggle_ecg_digitization) (MIT), commit `8358d5b`
- Weights and configs: [kaggle.com/datasets/dangnh0611/ecg-checkpoints](https://www.kaggle.com/datasets/dangnh0611/ecg-checkpoints)
- Inference notebook: [kaggle.com/code/dangnh0611/5th-place-solution](https://www.kaggle.com/code/dangnh0611/5th-place-solution)
- Write-up: [5th place solution: Multi-stages Heatmap-based Modeling](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/5th-place-solution)

### Pipeline

| Stage | Model | Input size |
|---|---|---|
| 1. Orientation correction | EfficientViT-B2 (predicts 0/90/180/270° rotation and the exact angle) | 512×512 |
| 2. Keypoint detection and image registration | UNet with ConvNeXt-small encoder, 5-fold ensemble | 2048×2048 |
| 3. Waveform estimation | 0.7 × CoaT-Lite-Medium+VGG19 and 0.3 × ConvNeXt-Large+VGG19 (dual encoder) | 1024×1024 |
| 4. Refinement for large amplitudes | "TALL" ConvNeXt-small, for signals near ±3.2 mV | 1024×512 |

## Candidates considered

Only solutions with public code and trained weights were considered. The 1st–3rd place teams have not published code.

| Candidate | Private LB SNR (dB) | Code | Weights | Inference code | Decision |
|---|---|---|---|---|---|
| **5th place, full ensemble** (dangnh0611) | **22.63** | ✅ MIT | ✅ | ✅ notebook + `submit.py` | **Selected** |
| 5th place, best single waveform model | 22.35 | ✅ | ✅ | needs config changes | Fallback if the ensemble is too slow to deploy |
| 6th place (GWwangshuo) | 22.25 (best single model) | ✅ (no license) | ✅ 0.7 GB | ✅ notebook | Rejected: lower score; stages 0–1 come from a separate public notebook |
| 4th place (uchiyama33) | n/a | ✅ (no license) | ❌ | ❌ | Rejected: we would have to train it ourselves |
| Train our own (#5) | n/a | n/a | n/a | n/a | Rejected: the goal is to skip training and focus on deployment |

Scores are from the authors' READMEs, on Kaggle's hidden private test set. That set was not used for training, so it is a fair comparison.

## Rationale

1. **It has the highest score among usable solutions.** At 22.63 dB, it beats the best 6th-place model by about 0.4 dB.
2. **We use it exactly as the authors submitted it.** Any change (fewer folds, dropping a model) would give a pipeline with no verified score. The 22.35 dB single-model variant is the fallback if speed or cost becomes a problem in deployment.
3. **It is the most complete package.** It has trained weights, configs, a Docker environment, inference code, ablation studies and data-processing docs, so it is the least work to integrate.
4. **It is known to run on a single GPU.** The submission ran inside Kaggle's GPU notebook limits.

## Known limitations

- **It needs a GPU.** It needs CUDA (PyTorch 2.6, CUDA 12.6 Docker image). It will not run in a practical way on our laptops (e.g. an M3 Mac with 16 GB and no CUDA). Run evaluation on Kaggle Notebooks or a cloud GPU.
- **Checkpoints are large.** The checkpoint dataset is about 19 GB. Only the subset used in the final submission needs to be downloaded.
- **Output range.** The main models cover ±3.2 mV. Larger amplitudes rely on the TALL refinement model.
- **It is fitted to the competition's images.** It expects 12 leads, a 10 s layout, and the competition's image types. Performance on other printers or phone photos is unknown until we test it (#14, #11).

## Test evaluation (held-out)

> **Status: blocked.** This needs a held-out test set and the official metric.

- **Test set:** Do **not** use the Kaggle `train` set. The final models were trained on all of it (`ALLDATA` in the checkpoint names), so scores on it would be inflated. Use the external datasets from #14, which the model has never seen.
- **Metric:** Use the official competition SNR: ±0.2 s time shift, free vertical offset, signal and noise power summed over all 12 leads per image, averaged in linear scale, then converted to dB. The `scripts/evaluate.py` from #7 must match this before it is used here.
- **Protocol:** Run the selected pipeline **once** on the test set and record the result below. Do not tune anything afterwards.

| Test set | Records | SNR (dB) | MAE (mV) | Date | Run link |
|---|---|---|---|---|---|
| _pending (#14)_ | | | | | |

## Registration

> **Status: blocked on #12** (model registry).

Register as the `production` alias. Include:
- the weights for every stage, with their Kaggle dataset version
- the upstream commit `8358d5b` and every stage config `.yaml`
- the Kaggle private LB score (22.63 dB) and the held-out test score from the table above
- a link to this document
