# Evaluation Metrics

Implementation: `src/ekg_digitizer/evaluation/`. Entry point: `scripts/evaluate.py`.

Each record is a set of leads (`I`, `II`, ..., `V6`). Each lead is a 1-D signal in mV at sampling frequency `fs`.
Predicted signals are scored against the reference signals.

## Metrics

| Metric | Definition | Better |
|---|---|---|
| **SNR** (primary) | `10·log10( Σ x_ref² / Σ (x_ref − x_est)² )` in dB | higher |
| **MAE** | `mean |x_ref − x_est|` in mV | lower |

Both are computed per lead, averaged over the leads in a record, then averaged over records.
SNR is the competition metric, so use it to compare models. MAE is in mV, which makes it easier to interpret.

### Alignment

Before scoring, each predicted lead is shifted in time and moved up or down to line up with the reference. This follows the [PhysioNet Challenge 2024 evaluation code](https://github.com/physionetchallenges/evaluation-2024):

1. The time shift is the lag within ±0.5 s with the highest Pearson correlation to the reference.
2. The vertical offset is the mean difference `x_ref − x_est` after shifting.
3. If that offset is larger than 1 mV, the prediction is scored unaligned.

> **TODO:** Confirm these limits against the Kaggle competition's evaluation page. If they differ, update `MAX_SHIFT_S` and `MAX_OFFSET_MV` in `metrics.py`.

### Missing samples

- Non-finite values (NaN or ±inf) mean "no prediction". A prediction shorter than the reference is padded with NaN.
- SNR is multiplied by the fraction of reference samples the prediction covers. A lead or record with no prediction scores 0 dB.

## Usage

Each record is a CSV with a header row of lead names and one column per lead. Reference and prediction files have the same name.

```bash
PYTHONPATH=src python scripts/evaluate.py --ref-dir data/val/signals --pred-dir outputs/val --fs 500
```
