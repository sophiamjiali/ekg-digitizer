# ==============================================================================
# File Name : register_model.py
# Purpose   : Registers a training run's model and promotes registered versions
# ==============================================================================
"""Register models in the W&B Model Registry and move the staging/production aliases.

Examples:
    # Register the model logged by a training run, marked for review
    python scripts/register_model.py register --run <entity>/<project>/<run_id> --alias staging

    # Approve version v3 for serving
    python scripts/register_model.py promote --version v3 --alias production
"""

import argparse

import wandb

from ekg_digitizer.registry import ALIASES, COLLECTION, collection_path, promote, register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=COLLECTION, help="Registry collection name")
    commands = parser.add_subparsers(dest="command", required=True)

    reg = commands.add_parser("register", help="Register the model logged by a training run")
    reg.add_argument("--run", required=True, help="W&B run path: <entity>/<project>/<run_id>")
    reg.add_argument("--alias", choices=ALIASES, help="Optional alias for the new version")

    pro = commands.add_parser("promote", help="Point an alias at a registered version")
    pro.add_argument("--version", required=True, help="Registered version, e.g. v3")
    pro.add_argument("--alias", required=True, choices=ALIASES)

    args = parser.parse_args()
    api = wandb.Api()

    if args.command == "register":
        artifact = register(api.run(args.run), alias=args.alias, collection=args.collection)
        print(f"Registered {args.run} as {collection_path(args.collection)}:{artifact.version}")
    else:
        promote(api, args.version, args.alias, collection=args.collection)
        print(f"{collection_path(args.collection)}:{args.version} is now '{args.alias}'")


if __name__ == "__main__":
    main()
