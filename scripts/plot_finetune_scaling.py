"""Draw the E3 data-scaling figure of FOUNDATION_FINETUNE.md from the committed results.

Reads ``results/finetune/finetune_metrics_merged5.json`` (no ground truth, no sibling
repo) and writes ``reports/finetune_data_scaling.png`` next to the report.

Usage::

    pixi run python scripts/plot_finetune_scaling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from object_detection_eval.report.finetune import load_finetune_results, render_scaling_png

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _REPO_ROOT / "benchmarks/basketball/results/finetune/finetune_metrics_merged5.json"
_OUT = _REPO_ROOT / "benchmarks/basketball/reports/finetune_data_scaling.png"


def main(argv: list[str] | None = None) -> int:
    """Render the PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=_RESULTS)
    parser.add_argument("--out", type=Path, default=_OUT)
    args = parser.parse_args(argv)
    render_scaling_png(load_finetune_results(args.results), args.out)
    logger.info("wrote {}", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
