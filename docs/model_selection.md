# Model Selection

Records which model is deployed and why. Update this file each time a new model is selected.

## Process

1. Produce test-split predictions for each candidate model with the inference pipeline:

   ```bash
   PYTHONPATH=src python scripts/inference.py --checkpoint <run_id>/best_model.pth \
       --image-dir data/test --split-csv data/test.csv --out-dir outputs/test/<run_id>
   ```

   Use the config the candidate was trained with (`--config`); otherwise the model won't load.
2. Score and register the best one (see `scripts/select_model.py`):

   ```bash
   PYTHONPATH=src python scripts/select_model.py --ref-dir data/test \
       --candidate <entity>/<project>/<run_id>=outputs/test/<run_id> --alias staging
   ```

   Use `--dry-run` to see the scores without registering anything.
3. Fill in the selection below, then promote the version once it's reviewed:
   `PYTHONPATH=src python scripts/register_model.py promote --version vN --alias production`

**The test split is used once per selection.** Compare and tune models on the validation split only. If test scores are used to make changes, the reported results are no longer an honest estimate.

SNR is the primary metric and MAE breaks ties (see [metrics.md](metrics.md)). Test scores are stored in the registered model's metadata under `test_metrics`.

## Current selection

The retrained checkpoint, after the decoder fix in `3c645ac` (the model keeps the image's horizontal axis instead of pooling it to a single vector). Trained with `configs/config.yaml` on `main`.

| Field | Value |
|---|---|
| Registry version | `wandb-registry-model/ekg-digitizer:v2` |
| Alias | `production` and `staging` are on `v2`; `v0`/`v1` (pre-fix model) have no aliases and remain available for rollback |
| W&B run | `stanleycao23-university-of-toronto/ekg-digitizer/q46r3hgv` |
| Deployed | Cloud Run release `v0.1.3` (the first release that includes the model fix; older releases can't run `v2` correctly) |
| Evaluated on | 2026-11-25, 8 training records (`tests/fixtures/regression_ids.txt`), **not** a held-out test split |

### Candidates

| Version | Run | SNR (dB) | MAE (mV) | Notes |
|---|---|---|---|---|
| `v2` | `q46r3hgv` | **1.592** | **0.0764** | Retrained after the decoder fix; epoch 45, training loss 0.128 |
| `v0` / `v1` | `7wv7g4ia` / `itiu5j1q` | 0.119 | 0.0873 | Pre-fix model; outputs a near-flat line for every lead |

All scores are on the same 8 training records, default config, TTA on.

### Rationale

- `v2` beats the previous model on both metrics (SNR +1.47 dB, MAE −12%) and produces waveforms that follow the reference, where `v0`/`v1` produced a flat line.
- Scores are stored in the registry metadata as `train_subset_snr` / `train_subset_mae`, not `test_metrics`, because the records were seen in training.

### Exception: no held-out test evaluation

The acceptance criterion "evaluated once on the held-out test split" is waived for this selection, by team decision. No held-out split with reference signals exists, so the 8-record training subset is the only evaluation available. The scores above therefore overstate how well the model generalizes.

If a held-out split becomes available, evaluate `v2` on it once (see Process above), store the result as `test_metrics`, and update this section.
