# ==============================================================================
# File Name : error_analysis.py
# Purpose   : Ranks validation records by score and plots the worst for review
# ==============================================================================
"""Find the worst-performing validation records and plot them for review.

Writes to ``--out-dir``:
    scores.csv   : every record ranked worst-first, with SNR, MAE and per-lead SNR
    worst/*.png  : the source image (if found) and each lead's reference vs.
                   prediction for the ``--top`` worst records

Example:
    python scripts/error_analysis.py --ref-dir data/val/signals --pred-dir outputs/val \
        --image-dir data/val/images --out-dir outputs/error_analysis --top 20
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no display needed
import matplotlib.pyplot as plt
import numpy as np

from ekg_digitizer.evaluation import align, load_split, score_record
from ekg_digitizer.evaluation.evaluate import Record, _fit_length
from ekg_digitizer.evaluation.metrics import LEADS, power, snr_ratio


def to_db(ratio: float) -> float:
    """Convert one linear SNR ratio to dB."""
    return float(10 * np.log10(max(ratio, 1e-12)))


def aligned_leads(reference: Record, prediction: Record) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{lead: (reference, aligned prediction)}``, aligned the same way the metric does."""
    fs = reference["II"].size / 10  # lead II spans 10 s
    pairs = {}
    for lead in (lead for lead in LEADS if lead in reference):
        x_ref = reference[lead]
        x_est = _fit_length(prediction.get(lead, np.zeros(0)), x_ref.size)
        pairs[lead] = (x_ref, align(x_ref, x_est, fs))
    return pairs


def score_row(record_id: str, reference: Record, prediction: Record) -> dict[str, float | str]:
    """Record-level SNR (dB) and MAE, plus each lead's SNR (dB) to show which leads fail."""
    ratio, record_mae = score_record(reference, prediction)
    row: dict[str, float | str] = {"record_id": record_id, "snr_db": to_db(ratio), "mae": record_mae}
    for lead, (x_ref, aligned) in aligned_leads(reference, prediction).items():
        row[f"snr_{lead}"] = to_db(snr_ratio(*power(x_ref, aligned)))
    return row


def find_image(image_dir: Path | None, record_id: str) -> Path | None:
    """First image whose name starts with the record id, e.g. ``<id>-0001.png``."""
    if image_dir is None:
        return None
    return next((p for p in sorted(image_dir.rglob(f"{record_id}*")) if p.suffix in {".png", ".jpg"}), None)


def plot_record(row: dict, reference: Record, prediction: Record, image: Path | None, out: Path) -> None:
    """Save the source image above a grid of reference (black) vs. prediction (red) per lead."""
    pairs = aligned_leads(reference, prediction)
    fig = plt.figure(figsize=(16, 14 if image else 9))
    grid = fig.add_gridspec(5 if image else 3, 4)

    if image:
        ax = fig.add_subplot(grid[:2, :])
        ax.imshow(plt.imread(image))
        ax.set_axis_off()

    for i, (lead, (x_ref, aligned)) in enumerate(pairs.items()):
        ax = fig.add_subplot(grid[(2 if image else 0) + i // 4, i % 4])
        ax.plot(x_ref, color="black", linewidth=0.7, label="reference")
        ax.plot(aligned, color="red", linewidth=0.7, alpha=0.8, label="prediction")
        ax.set_title(f"{lead}  ({row[f'snr_{lead}']:.1f} dB)", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"{row['record_id']}  SNR {row['snr_db']:.2f} dB  MAE {row['mae']:.3f} mV")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-dir", type=Path, required=True, help="Directory of reference record CSVs")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory of predicted record CSVs")
    parser.add_argument("--image-dir", type=Path, help="Directory of source EKG images (optional)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/error_analysis"))
    parser.add_argument("--top", type=int, default=20, help="Number of worst records to plot")
    args = parser.parse_args()

    references, predictions = load_split(args.ref_dir), load_split(args.pred_dir)

    # Rank worst-first; missing predictions score as all zeros, like evaluate.py
    rows = [score_row(rid, ref, predictions.get(rid, {})) for rid, ref in references.items()]
    rows.sort(key=lambda r: r["snr_db"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["record_id", "snr_db", "mae"] + [f"snr_{lead}" for lead in LEADS]
    with open(args.out_dir / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)

    worst_dir = args.out_dir / "worst"
    worst_dir.mkdir(exist_ok=True)
    for rank, row in enumerate(rows[: args.top], start=1):
        rid = row["record_id"]
        out = worst_dir / f"{rank:03d}_{rid}.png"
        plot_record(row, references[rid], predictions.get(rid, {}), find_image(args.image_dir, rid), out)

    print(f"Scored {len(rows)} records -> {args.out_dir / 'scores.csv'}")
    print(f"Plotted worst {min(args.top, len(rows))} -> {worst_dir}")


if __name__ == "__main__":
    main()
