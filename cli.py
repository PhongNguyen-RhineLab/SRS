"""
Shared command-line handling for all entry scripts.

Before this module existed, switching dataset meant editing a
`DATASET = "..."` constant at the top of every script (train_rl.py,
run_baselines.py, run_test_eval.py, ...) one by one. Now every entry
script does:

    from cli import build_config
    cfg = build_config("short description of the script")

and is run as, for example:

    python train_rl.py --dataset toys
    python run_baselines.py -d ml-1m --checkpoint-dir some/other/dir

Aliases accepted for --dataset (case-insensitive):

    amazon_beauty : beauty, amazon_beauty
    movielens_1m  : ml-1m, ml1m, movielens, movielens_1m
    amazon_sports : sports, sports_and_outdoors, amazon_sports
    amazon_toys   : toys, toys_and_games, amazon_toys

The default (no flag) is amazon_beauty, matching the paper.
"""

import argparse

from config import Config, DATASET_REGISTRY

ALIASES = {
    "beauty": "amazon_beauty",
    "amazon_beauty": "amazon_beauty",
    "ml-1m": "movielens_1m",
    "ml1m": "movielens_1m",
    "movielens": "movielens_1m",
    "movielens_1m": "movielens_1m",
    "sports": "amazon_sports",
    "sports_and_outdoors": "amazon_sports",
    "amazon_sports": "amazon_sports",
    "toys": "amazon_toys",
    "toys_and_games": "amazon_toys",
    "amazon_toys": "amazon_toys",
}


def resolve_dataset(name: str) -> str:
    key = name.strip().lower()
    if key not in ALIASES:
        known = ", ".join(sorted(set(ALIASES)))
        raise SystemExit(
            f"Unknown dataset '{name}'. Accepted values (aliases included): {known}"
        )
    return ALIASES[key]


def make_parser(description: str) -> argparse.ArgumentParser:
    """
    Base parser with the flags every entry script shares. Scripts that need
    extra flags (e.g. diagnose.py's --selftest) can add them onto the
    returned parser before calling parse.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "-d", "--dataset", default="amazon_beauty",
        help="Dataset to run on. One of: "
             + ", ".join(sorted(DATASET_REGISTRY))
             + " (aliases: beauty, ml-1m, sports, toys). Default: amazon_beauty.",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Override the dataset directory (default: per-dataset, see config.py).",
    )
    p.add_argument(
        "--checkpoint-dir", default=None,
        help="Override the checkpoint directory (default: per-dataset, see config.py). "
             "Useful if e.g. you trained ml-1m but saved into the plain 'checkpoints' folder.",
    )
    return p


def config_from_args(args) -> Config:
    kwargs = {"dataset": resolve_dataset(args.dataset)}
    if getattr(args, "data_dir", None):
        kwargs["data_dir"] = args.data_dir
    if getattr(args, "checkpoint_dir", None):
        kwargs["checkpoint_dir"] = args.checkpoint_dir
    return Config(**kwargs)


def build_config(description: str) -> Config:
    """One-liner used by scripts that need no extra flags."""
    return config_from_args(make_parser(description).parse_args())
