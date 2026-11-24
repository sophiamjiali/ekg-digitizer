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

The team's only trained checkpoint (`best_model.pth`, trained with `configs/config.yaml` on `main`).

| Field | Value |
|---|---|
| Registry version | `wandb-registry-model/ekg-digitizer:v0` (`v1` holds the same weights, added for the rollback test) |
| Alias | `production` is on `v1` for the rollback test; `v0` and `v1` serve identical predictions |
| W&B run | `stanleycao23-university-of-toronto/ekg-digitizer/7wv7g4ia` (`v1`: `.../itiu5j1q`) |
| Evaluated on | 2026-11-23, 8 training records (`tests/fixtures/regression_ids.txt`), **not** a held-out test split |

### Candidates

| Run | SNR (dB) | MAE (mV) | Notes (key config differences) |
|---|---|---|---|
| `7wv7g4ia` | 0.119 | 0.0873 | Only candidate; default config, TTA on. Scores are on 8 training records |

### Rationale

- It is the only trained model, so there was nothing to compare against.
- Registered and promoted so the serving path (registry -> Cloud Run) works end to end. Scores are stored in the registry metadata as `train_subset_snr` / `train_subset_mae`, not `test_metrics`, because the records were seen in training.
- 0.12 dB SNR is close to a flat-line prediction, so treat this as a placeholder rather than a model fit for use.

### Still to do

- Evaluate on a held-out test split (`data/test`, see Process above) once one exists, and store the result as `test_metrics`.
- Retrain or compare further candidates if the test SNR confirms the low score, then update this section.
