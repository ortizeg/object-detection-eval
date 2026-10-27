"""Turn the per-call usage logs into ``cost_per_frame.json``, ``COST.md`` and the PNG.

Reads, under ``benchmarks/basketball/results/cost/``:

- ``usage/<pipeline>__<run>.jsonl`` -- one record per billed request, written by
  ``scripts/cost_rerun.py`` through ``inference/vlm/usage.py``;
- ``runs/<pipeline>__<run>.json`` -- the run manifests (frames, workers, timings).

and the two committed configs ``conf/cost_prices.yaml`` (prices with effective dates and
source URLs) and ``conf/cost_published_accuracy.yaml`` (accuracy copied from the
reports, each value with its citation). Writes ``cost_per_frame.json`` (the only input
of the two renderers), ``reports/COST.md`` and ``results/cost/cost_vs_accuracy.png``.

``--check`` recomputes the JSON and the Markdown in memory from the committed inputs and
exits 1 on any byte difference (the PNG is excluded). It also re-verifies that every
published accuracy value still matches the file it cites.

Exit codes: 0 ok; 1 drift, a failed citation, or a refusal to aggregate; 2 no usage logs
yet (the paid reruns have not been run: see docs/plans/cost-per-frame-2026-10.md).

    pixi run python scripts/cost_report.py
    pixi run python scripts/cost_report.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from loguru import logger

from object_detection_eval.cost.compute import (
    build_report,
    load_accuracy,
    load_prices,
    load_runs,
    verify_citations,
)
from object_detection_eval.cost.report import render_markdown, render_png
from object_detection_eval.cost.schemas import CostReport

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASKETBALL = _REPO_ROOT / "benchmarks" / "basketball"
JSON_NAME = "cost_per_frame.json"
PNG_NAME = "cost_vs_accuracy.png"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build(results_dir: Path, prices_path: Path, accuracy_path: Path) -> CostReport:
    """Aggregate everything under ``results_dir`` into the report."""
    loaded = load_runs(results_dir)
    return build_report(
        loaded,
        load_prices(prices_path),
        load_accuracy(accuracy_path),
        price_path=_rel(prices_path),
        accuracy_path=_rel(accuracy_path),
        price_sha256=hashlib.sha256(prices_path.read_bytes()).hexdigest(),
        accuracy_sha256=hashlib.sha256(accuracy_path.read_bytes()).hexdigest(),
    )


def serialize(report: CostReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=_BASKETBALL / "results" / "cost")
    parser.add_argument("--prices", type=Path, default=_BASKETBALL / "conf" / "cost_prices.yaml")
    parser.add_argument(
        "--accuracy", type=Path, default=_BASKETBALL / "conf" / "cost_published_accuracy.yaml"
    )
    parser.add_argument("--report", type=Path, default=_BASKETBALL / "reports" / "COST.md")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed JSON or Markdown differs from what the inputs generate.",
    )
    args = parser.parse_args(argv)

    usage_dir = args.results_dir / "usage"
    if not usage_dir.is_dir() or not any(usage_dir.glob("*.jsonl")):
        logger.error(
            "No usage logs under {}. The paid reruns have not been run yet: follow the "
            "command sequence in docs/plans/cost-per-frame-2026-10.md.",
            usage_dir,
        )
        return 2

    accuracy = load_accuracy(args.accuracy)
    problems = verify_citations(accuracy, _REPO_ROOT)
    for problem in problems:
        logger.error("citation: {}", problem)
    if problems:
        return 1

    try:
        report = build(args.results_dir, args.prices, args.accuracy)
    except (ValueError, KeyError) as exc:
        logger.error("Refusing to aggregate: {}", exc)
        return 1

    json_text = serialize(report)
    json_path = args.results_dir / JSON_NAME
    png_path = args.results_dir / PNG_NAME
    figure_href = Path(os.path.relpath(png_path, args.report.parent)).as_posix()
    md_text = render_markdown(CostReport.model_validate_json(json_text), figure_href)

    if args.check:
        drifted = False
        for path, text in ((json_path, json_text), (args.report, md_text)):
            if not path.is_file():
                logger.error("{} is missing; run without --check first.", path)
                drifted = True
            elif path.read_text(encoding="utf-8") != text:
                logger.error("{} has drifted from its inputs; rerun without --check.", path)
                drifted = True
        if not drifted:
            logger.info("{} and {} match their inputs.", json_path, args.report)
        return 1 if drifted else 0

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(md_text, encoding="utf-8")
    # The Markdown and the PNG are both rendered from the parsed JSON, never from raw logs.
    skipped = render_png(CostReport.model_validate_json(json_text), png_path)
    for line in skipped:
        logger.warning("plot: {}", line)
    logger.info("Wrote {}, {} and {}", json_path, args.report, png_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
