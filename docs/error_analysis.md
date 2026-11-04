# Error Analysis (#11)

Where and why the model fails, to guide what to fix next.

## How to reproduce

Run inference on the validation split, then rank and plot the worst records:

```bash
PYTHONPATH=src python scripts/error_analysis.py \
    --ref-dir <val signals> --pred-dir <val predictions> \
    --image-dir <val images> --out-dir outputs/error_analysis --top 20
```

- `outputs/error_analysis/scores.csv`: every record ranked worst-first, with record SNR, MAE and per-lead SNR.
- `outputs/error_analysis/worst/*.png`: source image plus reference (black) vs. prediction (red) for each lead.

## Run details

| Model (registry version / run) | Validation split | Records | SNR (dB) | MAE (mV) | Date |
|---|---|---|---|---|---|
| TODO | TODO | | | | |

## Failure categories

Tag each of the worst records with one or more categories. Add categories as they come up.

| Category | What it looks like | Records | Count |
|---|---|---|---|
| Skewed / rotated scan | Image tilted; predictions drift across the strip | | |
| Noise / low quality | Blur, stains, low contrast, photo glare | | |
| Grid artifacts | Prediction follows grid lines instead of the trace | | |
| Overlapping leads | Large amplitudes cross into a neighbouring lead | | |
| Wrong lead / layout | Lead swapped or segment boundaries misplaced | | |
| Amplitude scale | Shape right, amplitude off (MAE high, SNR moderate) | | |
| Missing output | Lead predicted as flat or empty | | |

## Findings

TODO: summary of the dominant failure patterns and which leads are worst (see the per-lead SNR columns).

## Follow-up issues

| Finding | Issue |
|---|---|
| TODO | #… |
