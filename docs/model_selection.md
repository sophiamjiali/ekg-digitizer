# Model Selection

Records which model is deployed and why. Update this file each time a new model is selected.

## Process

1. Produce test-split predictions for each candidate model.
2. Score and register the best one (see `scripts/select_model.py`):

   ```bash
   PYTHONPATH=src python scripts/select_model.py --ref-dir data/test/signals \
       --candidate <entity>/<project>/<run_id>=outputs/test/<run_id> --alias staging
   ```

   Use `--dry-run` to see the scores without registering anything.
3. Fill in the selection below, then promote the version once it's reviewed:
   `PYTHONPATH=src python scripts/register_model.py promote --version vN --alias production`

**The test split is used once per selection.** Compare and tune models on the validation split only. If test scores are used to make changes, the reported results are no longer an honest estimate.

SNR is the primary metric and MAE breaks ties (see [metrics.md](metrics.md)). Test scores are stored in the registered model's metadata under `test_metrics`.

## Current selection

_Not yet selected: waiting on the inference pipeline (#33)._

| Field | Value |
|---|---|
| Registry version | `wandb-registry-model/ekg-digitizer:vN` |
| Alias | staging / production |
| W&B run | `<entity>/<project>/<run_id>` |
| Test split evaluated on | YYYY-MM-DD |

### Candidates

| Run | SNR (dB) | MAE (mV) | Notes (key config differences) |
|---|---|---|---|
| | | | |

### Rationale

_Why this model was chosen. Include trade-offs beyond the scores, e.g. model size, inference speed, and robustness on external data such as LUDB._
