# Evaluation Metrics

Implementation: `src/ekg_digitizer/evaluation/`. Entry point: `scripts/evaluate.py`.

Each record is a 12-lead EKG. Each lead is a 1-D signal in mV. Predicted signals are scored against the reference signals.

## Metrics

| Metric | Definition | Better |
|---|---|---|
| **SNR** (primary) | The Kaggle competition's official metric, in dB (below) | higher |
| **MAE** | `mean |x_ref − x_est|` in mV, averaged over leads, then records | lower |

SNR is used to compare models, and its scores can be compared directly with the Kaggle leaderboard. MAE is in mV, which makes errors easier to interpret.

### SNR

This matches the official competition metric exactly. `tests/test_metrics.py` checks it against the official examples.

1. **Align each lead.**
   - Shift the prediction in time, by up to ±0.2 s, to the position with the highest cross-correlation with the reference.
   - Then subtract the vertical offset that minimizes the squared error. There is no limit on this offset.
2. **Per record:** sum the signal power `Σ x_ref²` and the noise power `Σ (x_ref − x_est)²` over **all leads**, then take `ratio = signal / noise`. The ratio is capped at 10⁶.
3. **Over the split:** `SNR = 10 · log10(mean of ratios)`. The ratios are averaged before converting to dB. The minimum score is −384 dB.

### Details

- The sampling frequency is inferred from lead II, which always covers 10 s: `fs = len(II) / 10`.
- Samples shifted out of range by alignment count as 0. So do missing values, missing leads, and missing records.
- Predictions longer than the reference are cut to its length. Shorter ones are padded with zeros.

## Usage

Each record is a CSV with a header row of lead names and one column per lead. Empty cells are ignored, since leads other than II usually cover only 2.5 s. Reference and prediction files have the same name.

```bash
PYTHONPATH=src python scripts/evaluate.py --ref-dir data/val/signals --pred-dir outputs/val
```
