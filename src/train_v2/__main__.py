"""Train Tadpole, the diagram reader.

Usage (from src/, with the D:/envs/diagram Python):
    python -m train_v2                      # full run, live dashboard
    python -m train_v2 --smoke              # 8 steps + tiny eval, to check the pipeline
    python -m train_v2 --config other.yaml --no-dashboard
To open it in its own window use train_v2/launch.ps1.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    ap.add_argument("--no-dashboard", action="store_true", help="plain logs instead of the live UI")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*causal_conv1d.*")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    from .trainer import Trainer  # heavy imports after argument parsing

    trainer = Trainer(cfg, smoke=args.smoke, dashboard=not args.no_dashboard)
    if args.no_dashboard:
        import logging

        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
