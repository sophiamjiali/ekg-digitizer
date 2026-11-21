# ==============================================================================
# File Name : select_model.py
# Purpose   : Scores candidate models on the test split and registers the best
# ==============================================================================
"""Pick the best model by test-split SNR and register it.

Each candidate is a W&B training run plus a directory of that model's
predicted record CSVs for the test split. The highest SNR wins (MAE breaks
ties). Run this once per selection: the test split must not guide training.

Example:
    python scripts/select_model.py --ref-dir data/test/signals \\
        --candidate <entity>/<project>/<run_a>=outputs/test/run_a \\
        --candidate <entity>/<project>/<run_b>=outputs/test/run_b \\
        --alias staging
"""

import argparse
from pathlib import Path

import wandb

from ekg_digitizer.evaluation import evaluate, load_split
from ekg_digitizer.registry import ALIASES, REGISTRY_PATH, register


def parse_candidate(value: str) -> tuple[str, Path]:
    """Split ``<run_path>=<pred_dir>`` into its two parts."""
    run_path, sep, pred_dir = value.partition("=")
    if not sep or not run_path or not pred_dir:
        raise argparse.ArgumentTypeError(f"Expected <run_path>=<pred_dir>, got {value!r}")
    return run_path, Path(pred_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-dir", type=Path, required=True, help="Directory of test-split reference CSVs")
    parser.add_argument(
        "--candidate",
        type=parse_candidate,
        action="append",
        required=True,
        help="<entity>/<project>/<run_id>=<pred_dir>; repeat for each model",
    )
    parser.add_argument("--alias", choices=ALIASES, default="staging")
    parser.add_argument("--dry-run", action="store_true", help="Score and pick only; don't register")
    args = parser.parse_args()

    # References are loaded once and shared by every candidate
    references = load_split(args.ref_dir)
    results: dict[str, dict[str, float]] = {
        run_path: evaluate(references, load_split(pred_dir)) for run_path, pred_dir in args.candidate
    }

    # Higher SNR is better; on a tie, lower MAE wins
    best = max(results, key=lambda run_path: (results[run_path]["snr"], -results[run_path]["mae"]))

    print(f"{'Run':<50} {'SNR (dB)':>10} {'MAE (mV)':>10} {'Records':>8}")
    for run_path, scores in sorted(results.items(), key=lambda item: -item[1]["snr"]):
        marker = " *" if run_path == best else ""
        print(f"{run_path:<50} {scores['snr']:>10.3f} {scores['mae']:>10.4f} {scores['num_records']:>8}{marker}")
    print(f"\nBest: {best}")

    if args.dry_run:
        return

    model = register(wandb.Api().run(best), args.alias, test_metrics=results[best])
    print(f"Registered {best} as {REGISTRY_PATH}:{model.version} ('{args.alias}')")
    print("Record the rationale in docs/model_selection.md")


if __name__ == "__main__":
    main()
