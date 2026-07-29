"""REPRO-03 statistical gate: reproduce the anchor bootstrap CIs and the joint-best tie.

Runs the ported, seeded paired bootstrap (``run_bootstrap`` / ``build_report``,
:mod:`object_detection_eval.metrics.bootstrap`) over the 94-image basketball
test predictions and asserts two things against the source repo's published
results:

Check A -- the 7-model @640 anchor (``bootstrap_5c_test_7models.json``):
every model's mAP_50:95 point estimate and 95% CI reproduce within
``--tolerance``, and every rank-adjacent pair reproduces the anchor's recorded
``ci_excludes_zero`` (5 of the 6 pairs are significant; the anchor records
RTMDet-M vs DAMO-YOLO-M as a tie, so faithful reproduction must too -- the
original report's "every adjacent pair is significant" prose over-claimed).

Check B -- the joint-best headline tie: YOLOX-M @800 (0.723) vs YOLO26m @640
(0.716) reproduces as a statistical TIE (point_diff ~= +0.0073, CI ~=
[-0.0033, +0.0190], ``ci_excludes_zero`` False) -- see ``EVAL_REPORT.md``'s
"Corrected leaderboard" section.

Both checks share the same 94-image ground truth and n_boot=1000/seed=0 as
the anchor. The manifest (``benchmarks/basketball/conf/reproduction_640.yaml``,
04-01) supplies the correct-variant @640 predictions paths for Check A; the
@800 YOLOX-M predictions file is read directly for Check B only -- feeding
it into Check A would reproduce the exact variant mix-up ``docs/methodology.
md`` now documents as resolved.

Usage::

    pixi run python scripts/run_bootstrap_gate.py

NOT wired into pytest: it reads source-repo-only artifacts (stored
predictions, the anchor json, the basketball test GT) absent from CI. See
``tests/scripts/test_run_bootstrap_gate.py`` for the CI-safe offline coverage
of the pure tie/significance and anchor-tolerance helpers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import supervision as sv
import yaml
from loguru import logger
from pydantic import BaseModel

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.bootstrap import build_report, load_predictions, run_bootstrap

_DEFAULT_SOURCE_REPO = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training")
_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/reproduction_640.yaml")
_DEFAULT_N_BOOT = 1000
_DEFAULT_SEED = 0
_DEFAULT_TOLERANCE = 0.01

# Relative to --source-repo.
_ANCHOR_REPORT_REL_PATH = ".deploy_comparison/bootstrap_5c_test_7models.json"
_YOLOX_M_800_REL_PATH = (
    "eval_output/official_2026-07-13/YOLOX-M-800/merged5/predictions_yolox_test.json.gz"
)

# The joint-best headline result (EVAL_REPORT.md "Corrected leaderboard"):
# YOLOX-M @800 (72.3) vs YOLO26m @640 (71.6) is a statistical tie.
_TIE_MODEL_A = "YOLOX-M@800"
_TIE_MODEL_B = "YOLO26m"
_EXPECTED_TIE_POINT_DIFF = 0.0073
_EXPECTED_TIE_CI_LOWER = -0.0033
_EXPECTED_TIE_CI_UPPER = 0.0190

_METRIC = "mAP_50_95"


class ManifestEntry(BaseModel, frozen=True):
    """The subset of a reproduction_640.yaml entry this gate needs.

    Extra manifest fields (onnx, labels, input_size, ...) are ignored --
    this gate only ever scores stored predictions, never runs ONNX inference.
    """

    name: str
    root: str
    predictions: str
    predictions_root: str | None = None

    @property
    def resolved_predictions_root(self) -> str:
        """The root name the ``predictions`` path resolves against."""
        return self.predictions_root if self.predictions_root is not None else self.root


class Manifest(BaseModel, frozen=True):
    """The committed reproduction manifest: 7 models in published rank order."""

    models: list[ManifestEntry]


def load_manifest(path: Path) -> Manifest:
    """Load and validate the committed reproduction manifest."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Manifest.model_validate(raw)


def within_tolerance(measured: float, expected: float, tolerance: float) -> bool:
    """True if `measured` is within `tolerance` of `expected` (boundary passes)."""
    return abs(measured - expected) <= tolerance


def ci_excludes_zero(ci_lower: float, ci_upper: float) -> bool:
    """True if the CI does not straddle zero (a statistically significant difference)."""
    return bool(ci_lower > 0.0 or ci_upper < 0.0)


def is_tie(ci_lower: float, ci_upper: float) -> bool:
    """True if the CI straddles zero -- a statistical tie (not significant)."""
    return not ci_excludes_zero(ci_lower, ci_upper)


def write_bootstrap_results(path: Path, report: dict[str, Any]) -> None:
    """Persist Check A's build_report() dict to ``path`` as indented JSON.

    Pure and numpy-free: ``build_report`` already returns plain Python
    floats/bools/strings, so ``report`` (config + per_model + pairwise,
    including each pair's ``ci_excludes_zero`` bool) round-trips through
    ``json.dump`` / ``json.load`` unchanged. Creates parent dirs and logs
    the written path. This is the ONLY writer of the committed @640 bootstrap
    accuracy file -- Check B's @800 comparison is never written here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote bootstrap results (Check A @640 anchor) to {path}")


def anchor_matches(measured: dict[str, float], anchor: dict[str, float], tolerance: float) -> bool:
    """True if point_estimate, ci_2.5, and ci_97.5 all fall within `tolerance` of `anchor`."""
    return all(
        within_tolerance(measured[key], anchor[key], tolerance)
        for key in ("point_estimate", "ci_2.5", "ci_97.5")
    )


def _resolve_source_repo_root(source_repo: Path, root: str) -> Path:
    if root != "source_repo":
        msg = (
            "run_bootstrap_gate only resolves manifest root='source_repo' entries "
            f"(predictions-only reproduction); got root={root!r}"
        )
        raise ValueError(msg)
    return source_repo / ".deploy_comparison"


def _manifest_predictions_path(entry: ManifestEntry, source_repo: Path) -> Path:
    root_dir = _resolve_source_repo_root(source_repo, entry.resolved_predictions_root)
    return root_dir / entry.predictions


def _materialize_gz(path: Path, dest_dir: Path) -> Path:
    """Gunzip a `.json.gz` predictions file into `dest_dir`; pass plain `.json` through."""
    if path.suffix != ".gz":
        return path
    dest = dest_dir / path.with_suffix("").name
    with gzip.open(path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def _assert_preconditions(manifest: Manifest, source_repo: Path, data_root: Path) -> None:
    """Halt with a clear per-path message if any required artifact is missing."""
    missing: list[Path] = []

    anchor_path = source_repo / _ANCHOR_REPORT_REL_PATH
    if not anchor_path.is_file():
        missing.append(anchor_path)

    for entry in manifest.models:
        pred_path = _manifest_predictions_path(entry, source_repo)
        if not pred_path.is_file():
            missing.append(pred_path)

    yolox_800_path = source_repo / _YOLOX_M_800_REL_PATH
    if not yolox_800_path.is_file():
        missing.append(yolox_800_path)

    gt_path = data_root / "test" / "_annotations.coco.json"
    if not gt_path.is_file():
        missing.append(gt_path)

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        msg = (
            "run_bootstrap_gate: required source-repo artifacts are missing "
            f"(precondition not met):\n{missing_list}"
        )
        raise FileNotFoundError(msg)


def _load_anchor(source_repo: Path) -> dict[str, Any]:
    anchor_path = source_repo / _ANCHOR_REPORT_REL_PATH
    with open(anchor_path) as f:
        anchor: dict[str, Any] = json.load(f)
    return anchor


def _run_check_a(
    manifest: Manifest,
    source_repo: Path,
    gt_map: dict[str, sv.Detections],
    n_boot: int,
    seed: int,
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    """Reproduce the 7-model @640 anchor: per-model CIs + adjacent-pair significance.

    Returns ``(passed, report)`` where ``report`` is the full
    :func:`~object_detection_eval.metrics.bootstrap.build_report` dict for
    the 7-model @640 anchor -- surfaced so ``main()`` can persist it behind
    ``--write-results`` (the @640 per_model CIs + pairwise significance,
    including the RTMDet-M vs DAMO-YOLO-M tie; 5 of 6 adjacent pairs
    significant). The joint-best headline tie is a SEPARATE Check B (@800
    YOLOX-M vs @640 YOLO26m) and is not part of this report.
    """
    pred_maps: dict[str, dict[str, sv.Detections]] = {}
    for entry in manifest.models:
        pred_path = _manifest_predictions_path(entry, source_repo)
        pred_maps[entry.name] = load_predictions(pred_path)

    logger.info(f"Check A: bootstrapping {len(pred_maps)} models (n_boot={n_boot}, seed={seed})")
    boot = run_bootstrap(gt_map, pred_maps, n_boot, seed)
    report = build_report(gt_map, pred_maps, boot, n_boot, seed)

    anchor = _load_anchor(source_repo)
    anchor_per_model = anchor["per_model"]

    header = (
        f"{'Model':<14} | {'Anchor pt':>10} | {'Measured pt':>11} | "
        f"{'Anchor CI':>22} | {'Measured CI':>22} | {'Within tol':>10}"
    )
    logger.info("=" * len(header))
    logger.info("Check A -- 7-model @640 anchor reproduction (bootstrap_5c_test_7models.json)")
    logger.info(header)
    logger.info("-" * len(header))

    all_within = True
    for entry in manifest.models:
        measured = report["per_model"][entry.name][_METRIC]
        anchor_stats = anchor_per_model[entry.name][_METRIC]
        ok = anchor_matches(measured, anchor_stats, tolerance)
        all_within = all_within and ok
        logger.info(
            f"{entry.name:<14} | {anchor_stats['point_estimate']:>10.4f} | "
            f"{measured['point_estimate']:>11.4f} | "
            f"[{anchor_stats['ci_2.5']:>8.4f}, {anchor_stats['ci_97.5']:>8.4f}] | "
            f"[{measured['ci_2.5']:>8.4f}, {measured['ci_97.5']:>8.4f}] | "
            f"{'yes' if ok else 'NO':>10}"
        )
    logger.info("=" * len(header))

    names_in_rank_order = [entry.name for entry in manifest.models]
    all_pairs_match = True
    logger.info(
        "Adjacent-pair significance (each pair must REPRODUCE the anchor's recorded "
        "ci_excludes_zero -- NOT a blanket 'all significant': the anchor itself records "
        "RTMDet-M vs DAMO-YOLO-M as a tie, so faithful reproduction must too):"
    )
    for higher, lower in pairwise(names_in_rank_order):
        key = f"{higher} minus {lower}"
        measured_pair = report["pairwise"][key][_METRIC]
        anchor_pair = anchor["pairwise"][key][_METRIC]
        measured_sig = ci_excludes_zero(measured_pair["ci_2.5"], measured_pair["ci_97.5"])
        anchor_sig = bool(anchor_pair["ci_excludes_zero"])
        ok = measured_sig == anchor_sig
        all_pairs_match = all_pairs_match and ok
        logger.info(
            f"  {higher} minus {lower}: diff={measured_pair['point_diff']:.4f} "
            f"CI=[{measured_pair['ci_2.5']:.4f}, {measured_pair['ci_97.5']:.4f}] "
            f"significant={'yes' if measured_sig else 'NO'} "
            f"anchor={'yes' if anchor_sig else 'NO'} reproduced={'yes' if ok else 'NO'}"
        )

    passed = all_within and all_pairs_match
    logger.info(f"Check A {'PASSED' if passed else 'FAILED'}")
    return passed, report


def _run_check_b(
    manifest: Manifest,
    source_repo: Path,
    gt_map: dict[str, sv.Detections],
    n_boot: int,
    seed: int,
    tolerance: float,
    tmp_dir: Path,
) -> bool:
    """Reproduce the joint-best headline tie: YOLOX-M @800 vs YOLO26m @640."""
    yolo26m_entry = next(e for e in manifest.models if e.name == "YOLO26m")
    yolo26m_path = _manifest_predictions_path(yolo26m_entry, source_repo)
    yolox_800_path = _materialize_gz(source_repo / _YOLOX_M_800_REL_PATH, tmp_dir)

    pred_maps: dict[str, dict[str, sv.Detections]] = {
        _TIE_MODEL_A: load_predictions(yolox_800_path),
        _TIE_MODEL_B: load_predictions(yolo26m_path),
    }

    logger.info(f"Check B: bootstrapping the joint-best pair (n_boot={n_boot}, seed={seed})")
    boot = run_bootstrap(gt_map, pred_maps, n_boot, seed)
    report = build_report(gt_map, pred_maps, boot, n_boot, seed)

    pair = report["pairwise"][f"{_TIE_MODEL_A} minus {_TIE_MODEL_B}"][_METRIC]
    diff_ok = within_tolerance(pair["point_diff"], _EXPECTED_TIE_POINT_DIFF, tolerance)
    lower_ok = within_tolerance(pair["ci_2.5"], _EXPECTED_TIE_CI_LOWER, tolerance)
    upper_ok = within_tolerance(pair["ci_97.5"], _EXPECTED_TIE_CI_UPPER, tolerance)
    tie = is_tie(pair["ci_2.5"], pair["ci_97.5"])

    header = f"{'Comparison':<28} | {'Expected':>22} | {'Measured':>22} | {'Verdict':>10}"
    logger.info("=" * len(header))
    logger.info("Check B -- joint-best headline tie (YOLOX-M @800 vs YOLO26m @640)")
    logger.info(header)
    logger.info(
        f"{'point_diff':<28} | {_EXPECTED_TIE_POINT_DIFF:>22.4f} | "
        f"{pair['point_diff']:>22.4f} | {'yes' if diff_ok else 'NO':>10}"
    )
    logger.info(
        f"{'ci_2.5':<28} | {_EXPECTED_TIE_CI_LOWER:>22.4f} | "
        f"{pair['ci_2.5']:>22.4f} | {'yes' if lower_ok else 'NO':>10}"
    )
    logger.info(
        f"{'ci_97.5':<28} | {_EXPECTED_TIE_CI_UPPER:>22.4f} | "
        f"{pair['ci_97.5']:>22.4f} | {'yes' if upper_ok else 'NO':>10}"
    )
    logger.info("=" * len(header))

    passed = diff_ok and lower_ok and upper_ok and tie
    verdict = "TIE (not significant)" if tie else "SIGNIFICANT (unexpected)"
    logger.info(
        f"Delta = {pair['point_diff'] * 100:+.2f}pt, "
        f"CI = [{pair['ci_2.5'] * 100:+.2f}, {pair['ci_97.5'] * 100:+.2f}]pt -> {verdict}"
    )
    logger.info(f"Check B {'PASSED' if passed else 'FAILED'}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REPRO-03 statistical gate: reproduce the published 7-model "
            "@640 bootstrap anchor and the YOLOX-M@800-vs-YOLO26m joint-best "
            "tie via the ported seeded paired bootstrap."
        )
    )
    parser.add_argument("--source-repo", type=Path, default=_DEFAULT_SOURCE_REPO)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--n-boot", type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    parser.add_argument(
        "--write-results",
        type=Path,
        default=None,
        help=(
            "If set, serialize Check A's 7-model @640 build_report() output "
            "(config + per_model CIs + pairwise significance) to this JSON "
            "path. Only Check A's @640 report is written; Check B's @800 "
            "joint-best comparison is never persisted here. Persistence only "
            "-- does not change the gate verdict, tolerances, seed, or n_boot."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    _assert_preconditions(manifest, args.source_repo, args.data_root)

    name_to_id, _id_to_name = resolve_taxonomy("merged5")
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt_map = load_coco_gt(gt_path, name_to_id)
    logger.info(f"Loaded {len(gt_map)} ground-truth images from {gt_path}")

    check_a_passed, check_a_report = _run_check_a(
        manifest, args.source_repo, gt_map, args.n_boot, args.seed, args.tolerance
    )

    if args.write_results is not None:
        write_bootstrap_results(args.write_results, check_a_report)

    with tempfile.TemporaryDirectory(prefix="run_bootstrap_gate_") as tmp:
        check_b_passed = _run_check_b(
            manifest,
            args.source_repo,
            gt_map,
            args.n_boot,
            args.seed,
            args.tolerance,
            Path(tmp),
        )

    if not (check_a_passed and check_b_passed):
        logger.error("run_bootstrap_gate FAILED")
        sys.exit(1)

    logger.info("run_bootstrap_gate PASSED -- anchor CIs and joint-best tie both reproduce")


if __name__ == "__main__":
    main()
