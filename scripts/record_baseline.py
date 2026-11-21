# ==============================================================================
# File Name : record_baseline.py
# Purpose   : Records the SNR/MAE baseline used by tests/test_regression.py
# ==============================================================================
"""Score the checkpoint on the fixed regression records and save the baseline.

Run once after the checkpoint or the preprocessing changes on purpose. The
output file is what tests/test_regression.py compares new runs against.

Example (from the repo root):
    python scripts/record_baseline.py
"""

import json
from pathlib import Path

import pandas as pd

from ekg_digitizer.config import load_config
from ekg_digitizer.evaluation.evaluate import evaluate, load_record
from ekg_digitizer.inference import predict_record
from ekg_digitizer.models.dataset import EKGDataset

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "train"
CHECKPOINT = REPO / "best_model.pth"
IDS_FILE = REPO / "tests" / "fixtures" / "regression_ids.txt"
BASELINE_FILE = REPO / "tests" / "fixtures" / "regression_baseline.json"


def main() -> None:
    # Same ID list the regression test uses, so the two stay in sync
    ids = [
        line.strip()
        for line in IDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    config = load_config(str(REPO / "configs" / "config.yaml"))
    dataset = EKGDataset(
        images=pd.DataFrame({"id": ids}),
        data_dir=str(DATA_DIR),
        config=config,
        transform=None,
        is_train=True,
    )

    # Reference signals come from the CSVs; predictions come from the checkpoint
    references = {}
    predictions = {}
    for base_id in dataset.images["id"]:
        references[base_id] = load_record(DATA_DIR / str(base_id) / f"{base_id}.csv")
        predictions[base_id] = predict_record(str(CHECKPOINT), dataset.load_image(base_id))

    scores = evaluate(references, predictions)
    baseline = {"snr": scores["snr"], "mae": scores["mae"], "num_records": scores["num_records"]}

    BASELINE_FILE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {BASELINE_FILE}: SNR {baseline['snr']:.3f} dB, MAE {baseline['mae']:.4f} mV")


if __name__ == "__main__":
    main()
