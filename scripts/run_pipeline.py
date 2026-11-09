#!/usr/bin/env python
"""Command-line entry point.

Examples
    python scripts/run_pipeline.py --demo                   # synthetic end-to-end test
    python scripts/run_pipeline.py --steps download         # fetch ERA5-Land from CDS
    python scripts/run_pipeline.py                          # prepare + RQ1-RQ3 + tables
    python scripts/run_pipeline.py --steps rq3              # one step only
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esilflood.config import load_config  # noqa: E402
from esilflood.pipeline import DEFAULT_STEPS, STEPS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=None,
                    help=f"steps to run (default: {' '.join(DEFAULT_STEPS)})")
    ap.add_argument("--demo", action="store_true",
                    help="generate synthetic inputs in demo/ and run every step on them")
    ap.add_argument("--force", action="store_true", help="recompute processed forcing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.demo:
        from esilflood.synthetic import make_demo_inputs

        demo = ROOT / "demo"
        if demo.exists():
            shutil.rmtree(demo)
        logging.info("Generating synthetic demo inputs in %s (NOT research results)", demo)
        make_demo_inputs(demo)
        cfg = load_config(args.config, root=demo)
        cfg.raw["static"]["target_resolution_m"] = 250     # faster demo
        cfg.raw["project"]["bootstrap_iterations"] = 300
    else:
        cfg = load_config(args.config)

    cfg.ensure_dirs()
    for step in args.steps or DEFAULT_STEPS:
        logging.info("=== step: %s ===", step)
        if step == "prepare":
            STEPS[step](cfg, force=args.force)
        else:
            STEPS[step](cfg)
    logging.info("Tables in %s, figures in %s", cfg.tables, cfg.figures)


if __name__ == "__main__":
    main()
