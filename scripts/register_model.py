# ==============================================================================
# File Name : register_model.py
# Purpose   : Registers a training run's model and promotes registered versions
# ==============================================================================
"""Register models and move the staging/production aliases.

Examples:
    python scripts/register_model.py register --run <entity>/<project>/<run_id> --alias staging
    python scripts/register_model.py promote --version v3 --alias production
"""

import argparse

import wandb

from ekg_digitizer.registry import ALIASES, REGISTRY_PATH, promote, register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    reg = commands.add_parser("register", help="Register the model saved by a training run")
    reg.add_argument("--run", required=True, help="W&B run path: <entity>/<project>/<run_id>")
    reg.add_argument("--alias", choices=ALIASES)

    pro = commands.add_parser("promote", help="Move an alias to a registered version")
    pro.add_argument("--version", required=True, help="e.g. v3")
    pro.add_argument("--alias", required=True, choices=ALIASES)

    args = parser.parse_args()
    api = wandb.Api()

    if args.command == "register":
        model = register(api.run(args.run), args.alias)
        print(f"Registered {args.run} as {REGISTRY_PATH}:{model.version}")
    else:
        promote(api, args.version, args.alias)
        print(f"{REGISTRY_PATH}:{args.version} is now '{args.alias}'")


if __name__ == "__main__":
    main()
