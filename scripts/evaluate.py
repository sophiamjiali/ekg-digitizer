# ==============================================================================
# File Name : evaluate.py
# Purpose   : Command-line entry point for scoring predictions on a data split
# ==============================================================================
"""Report SNR and MAE of predicted signals on a validation split.

Example:
    python scripts/evaluate.py --ref-dir data/val/signals --pred-dir outputs/val
"""

import argparse
from pathlib import Path

from ekg_digitizer.evaluation import evaluate, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-dir", type=Path, required=True, help="Directory of reference record CSVs")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory of predicted record CSVs")
    args = parser.parse_args()

    scores = evaluate(load_split(args.ref_dir), load_split(args.pred_dir))
    print(f"Records  : {scores['num_records']}")
    print(f"SNR (dB) : {scores['snr']:.3f}")
    print(f"MAE (mV) : {scores['mae']:.4f}")


if __name__ == "__main__":
    main()
