# ==============================================================================
# File Name : inference.py
# Purpose   : Performs inference using the trained model
# Test
# ==============================================================================

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from pathlib import Path

import pandas as pd
import torch

from ekg_digitizer.config import load_config
from ekg_digitizer.inference import (
    default_device,
    index_images,
    load_model,
    predict_image,
    read_image,
    resample_record,
    write_record,
)


def main() -> None:

    parser = ArgumentParser(
        description     = __doc__, 
        formatter_class = RawDescriptionHelpFormatter
    )

    # Initialize all entry-point arguments
    parser.add_argument("--image-dir", type = Path, required = True, 
                        help = "Directory of EKG images")
    parser.add_argument("--out-dir", type = Path, required = True, 
                        help="Directory to write predicted record CSVs to")
    parser.add_argument("--checkpoint", type = Path, 
                        default = Path("best_model.pth"), 
                        help = "Training checkpoint or weights file")
    parser.add_argument("--config", type = Path, 
                        default = Path("configs/config.yaml"))
    parser.add_argument("--split-csv", type = Path, 
                        help = "CSV with an 'id' column and optionally 'fs'")
    parser.add_argument("--tta", action = BooleanOptionalAction, 
                        default = None, help = "Override augmentations.use_tta")
    parser.add_argument("--device", help = "Torch device (default: cuda if available, else cpu)")
    args = parser.parse_args()
 
    # Load the configurations and device from the arguments
    config  = load_config(str(args.config))
    base_fs = config.signal_properties.base_fs
    device  = torch.device(args.device) if args.device else default_device()
 
    images = index_images(args.image_dir)
    fs_by_id: dict[str, int] = {}
    if args.split_csv:
        split = pd.read_csv(args.split_csv, dtype={"id": str})
        ids   = split["id"].tolist()
        if "fs" in split: fs_by_id = dict(zip(ids, split["fs"].astype(int)))

    else: ids = sorted(images)
 
    missing = [record_id for record_id in ids if record_id not in images]
    if missing: parser.error(f"No image under {args.image_dir} for "
                             f"{len(missing)} record(s), e.g. {missing[:5]}")
 
    model = load_model(args.checkpoint, config, device)
    for n, record_id in enumerate(ids, start =1 ):

        record = predict_image(
            model   = model,
            image   = read_image(images[record_id]), 
            config  = config, 
            device  = device, 
            use_tta = args.tta
        )
        record = resample_record(
            record  = record, 
            fs      = fs_by_id.get(record_id, base_fs), 
            base_fs = base_fs
        )
        write_record(
            record = record, 
            path = args.out_dir / f"{record_id}.csv"
        )

        if n % 50 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} records")
 
    print(f"Wrote {len(ids)} records -> {args.out_dir}")


if __name__ == '__main__':
    main()