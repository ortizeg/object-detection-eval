"""Rerun one zero-shot API pipeline on the committed validation frames and log its usage.

The paid half of the cost-per-frame study (plan of record:
``docs/plans/cost-per-frame-2026-10.md``). Subcommands::

    select-frames [--data-root D] [--check]
        Write (or verify) conf/cost_val_frames.json: 20 validation frames, seed 20261027,
        stratified across the validation clips, ordered round-robin so any prefix spans
        every clip. Reads only the validation annotations.

    check-config
        Resolve all 8 pipelines offline (no key, no network, nothing constructed) and check
        the pipeline / fused-line ids against conf/cost_published_accuracy.yaml.

    run --pipeline ID --run {1,2,proj} [--max-frames N] [--dry-run] [--force] [--data-root D]
        One pipeline x one run at its published config and its published --workers, with
        every response cache and checkpoint bypassed. Writes
        results/cost/usage/ID__RUN.jsonl and results/cost/runs/ID__RUN.json. ``--run proj``
        is the 3-frame projection run and requires ``--max-frames``.

    project
        Project the study's spend from the ID__proj runs and compare it with the 40 USD cap.

Guards: there is no split option and no --workers option (both come from the committed
config); frames must all be validation frames and none may appear in the test annotations;
existing outputs are never overwritten without --force.

Exit codes:
    0  ok / run complete / projection within the cap
    1  check failed or an unexpected error
    2  project: a pipeline's projection files are missing
    3  project: the projected spend exceeds the cap (STOP and ask the user)
    4  run aborted (a call failed); inspect, then rerun that (pipeline, run) with --force
    5  run refused (bad frame, existing outputs, unknown pipeline, bad arguments)

The provider key is read from the environment by the inferencers: OPENAI_KEY or
OPENAI_API_KEY (Astra, Sol), ANTHROPIC_KEY or ANTHROPIC_API_KEY (Opus), GEMINI_API_KEY or
GOOGLE_API_KEY (Gemini; set GEMINI_API_KEY="$GEMINI_KEY"), DINOX_KEY or DDS_API_TOKEN (DINO-X).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

from object_detection_eval.cost.compute import load_accuracy, load_prices, load_runs, project
from object_detection_eval.cost.frames import select_stratified
from object_detection_eval.cost.rerun import (
    FrameRefusalError,
    RerunConfig,
    annotation_filenames,
    build_predictor,
    check_against_accuracy,
    load_frames_file,
    load_rerun_config,
    resolve_pipeline,
    run_frames,
    sha256_file,
    validate_frames,
)
from object_detection_eval.data.clips import clip_key

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONF = _REPO_ROOT / "benchmarks" / "basketball" / "conf"
_RESULTS = _REPO_ROOT / "benchmarks" / "basketball" / "results" / "cost"
_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_CONFIG_PATH = _CONF / "cost_rerun.yaml"
_FRAMES_PATH = _CONF / "cost_val_frames.json"
SEED = 20261027
N_FRAMES = 20

EXIT_OK, EXIT_FAIL, EXIT_MISSING, EXIT_OVER_CAP, EXIT_ABORTED, EXIT_REFUSED = 0, 1, 2, 3, 4, 5


# ----------------------------------------------------------------------- select-frames


def _build_frames_payload(data_root: Path) -> dict[str, object]:
    from object_detection_eval.cost.frames import largest_remainder

    annotations = data_root / "valid" / "_annotations.coco.json"
    names = sorted(annotation_filenames(data_root, "valid"))
    sizes: dict[str, int] = {}
    for name in names:
        key = clip_key(name)
        sizes[key] = sizes.get(key, 0) + 1
    return {
        "split": "valid",
        "seed": SEED,
        "n": N_FRAMES,
        "annotations_sha256": sha256_file(annotations),
        "clip_sizes": dict(sorted(sizes.items())),
        "allocation": dict(sorted(largest_remainder(sizes, N_FRAMES).items())),
        "frames": select_stratified(names, N_FRAMES, SEED),
    }


def cmd_select_frames(args: argparse.Namespace) -> int:
    payload = _build_frames_payload(args.data_root)
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.check:
        if not _FRAMES_PATH.is_file():
            logger.error("{} is missing; run select-frames without --check.", _FRAMES_PATH)
            return EXIT_FAIL
        if _FRAMES_PATH.read_text() != rendered:
            logger.error("{} does not match the local dataset's selection.", _FRAMES_PATH)
            return EXIT_FAIL
        logger.info("{} matches the dataset at {}", _FRAMES_PATH, args.data_root)
        return EXIT_OK
    _FRAMES_PATH.write_text(rendered)
    logger.info(
        "Wrote {} ({} frames, allocation {})", _FRAMES_PATH, N_FRAMES, payload["allocation"]
    )
    return EXIT_OK


# ------------------------------------------------------------------------ check-config


def _load_config() -> RerunConfig:
    return load_rerun_config(_CONFIG_PATH)


def cmd_check_config(_args: argparse.Namespace) -> int:
    config = _load_config()
    problems = check_against_accuracy(config, load_accuracy(_CONF / "cost_published_accuracy.yaml"))
    for spec in config.pipelines:
        try:
            resolved = resolve_pipeline(spec, _REPO_ROOT)
        except (ValueError, KeyError) as exc:
            problems.append(f"{spec.id}: {exc}")
            continue
        logger.info(
            "{:<24} env={:<10} workers={} {}",
            resolved.id,
            resolved.pixi_env,
            resolved.workers,
            {k: v for k, v in resolved.description.items() if k != "classes"},
        )
    frames = load_frames_file(_FRAMES_PATH)
    if frames.split != "valid" or len(set(frames.frames)) != frames.n:
        problems.append(f"{_FRAMES_PATH.name}: not a valid {frames.n} unique-frame val list")
    for problem in problems:
        logger.error(problem)
    if problems:
        return EXIT_FAIL
    logger.info(
        "check-config: {} pipelines resolve; ids match the accuracy config.", len(config.pipelines)
    )
    return EXIT_OK


# ------------------------------------------------------------------------------- run


def _outputs(pipeline: str, run: str) -> tuple[Path, Path]:
    return (
        _RESULTS / "usage" / f"{pipeline}__{run}.jsonl",
        _RESULTS / "runs" / f"{pipeline}__{run}.json",
    )


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config()
    try:
        spec = config.pipeline(args.pipeline)
    except KeyError as exc:
        logger.error("{}", exc.args[0])
        return EXIT_REFUSED

    if args.run == "proj" and args.max_frames is None:
        logger.error("--run proj requires --max-frames (the projection run is short by design).")
        return EXIT_REFUSED
    if args.run != "proj" and args.max_frames is not None:
        logger.error(
            "--max-frames is only for --run proj; runs 1 and 2 use all {} frames.", N_FRAMES
        )
        return EXIT_REFUSED
    if args.max_frames is not None and args.max_frames < 1:
        logger.error("--max-frames must be at least 1.")
        return EXIT_REFUSED

    frames_file = load_frames_file(_REPO_ROOT / config.frames_file)
    frames = frames_file.frames[: args.max_frames] if args.max_frames else list(frames_file.frames)
    try:
        validate_frames(frames, args.data_root)
    except (FrameRefusalError, FileNotFoundError) as exc:
        logger.error("Refusing to run: {}", exc)
        return EXIT_REFUSED

    usage_path, manifest_path = _outputs(spec.id, args.run)
    existing = [p for p in (usage_path, manifest_path) if p.exists()]
    if existing and not args.force:
        logger.error("Refusing to overwrite {} (use --force after inspecting it).", existing)
        return EXIT_REFUSED

    env = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if env is not None and env != spec.pixi_env:
        logger.warning(
            "{} is meant for the {!r} pixi environment but this is {!r}.",
            spec.id,
            spec.pixi_env,
            env,
        )

    try:
        resolved = resolve_pipeline(spec, _REPO_ROOT)
    except (ValueError, KeyError) as exc:
        logger.error("Published config no longer matches the plan of record: {}", exc)
        return EXIT_REFUSED

    if args.dry_run:
        logger.info("DRY RUN (nothing is constructed, nothing is called)")
        logger.info("pipeline : {} ({})", resolved.id, resolved.label)
        logger.info("env      : {}", resolved.pixi_env)
        logger.info(
            "workers  : {}  [{}]", resolved.workers, " ".join(resolved.workers_citation.split())
        )
        logger.info("config   : {}", json.dumps(resolved.description, indent=2, default=str))
        logger.info("frames   : {} (run {}) {}", len(frames), args.run, frames[:3])
        logger.info("outputs  : {} and {}", usage_path, manifest_path)
        return EXIT_OK

    for path in existing:  # --force: start the log clean so lines are never appended twice
        path.unlink()
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running {} run {}: {} frames, workers={}, caches bypassed",
        spec.id,
        args.run,
        len(frames),
        spec.workers,
    )
    predictor = build_predictor(spec, _REPO_ROOT)
    manifest = run_frames(
        predictor,
        frames,
        args.data_root / "valid",
        pipeline=spec.id,
        run=args.run,
        workers=spec.workers,
        workers_citation=" ".join(spec.workers_citation.split()),
        usage_path=usage_path,
        frames_file_sha256=sha256_file(_REPO_ROOT / config.frames_file),
        repo_root=_REPO_ROOT,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    n_lines = (
        sum(1 for line in usage_path.read_text().splitlines() if line.strip())
        if usage_path.exists()
        else 0
    )
    logger.info(
        "{} run {}: {} in {:.1f}s, {} usage records -> {}",
        spec.id,
        args.run,
        manifest.status,
        manifest.wall_s,
        n_lines,
        usage_path,
    )
    if manifest.status != "complete":
        logger.error("ABORTED: {}", manifest.abort_reason)
        return EXIT_ABORTED
    return EXIT_OK


# ---------------------------------------------------------------------------- project


def cmd_project(_args: argparse.Namespace) -> int:
    config = _load_config()
    frames_file = load_frames_file(_REPO_ROOT / config.frames_file)
    runs = {r.pipeline: r for r in load_runs(_RESULTS) if r.run == "proj"}
    missing = [p.id for p in config.pipelines if p.id not in runs or runs[p.id].manifest is None]
    if missing:
        logger.error("Missing projection runs for: {}", missing)
        return EXIT_MISSING
    projection = project(
        {pid: runs[pid] for pid in (p.id for p in config.pipelines)},
        {p.id: p.label for p in config.pipelines},
        load_prices(_CONF / "cost_prices.yaml"),
        budget_usd=config.budget_usd,
        frames_per_run=len(frames_file.frames),
        runs=config.runs,
    )
    out = _RESULTS / "projection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(projection.model_dump_json(indent=2) + "\n")
    logger.info(
        "{:<24} {:>7} {:>11} {:>14} {:>9}",
        "pipeline",
        "calls",
        "USD/frame",
        "remaining USD",
        "calls left",
    )
    for row in projection.rows:
        usd = "unpriced" if row.unpriced else f"{row.usd_per_frame_mean:.4f}"
        rem = "n/a" if row.projected_remaining_usd is None else f"{row.projected_remaining_usd:.2f}"
        logger.info(
            "{:<24} {:>7} {:>11} {:>14} {:>9}",
            row.pipeline,
            row.n_calls,
            usd,
            rem,
            row.projected_remaining_calls,
        )
    logger.info(
        "spent {:.2f} + remaining {:.2f} = projected {:.2f} USD against a {:.0f} USD cap -> {}",
        projection.spent_usd,
        projection.projected_remaining_usd,
        projection.projected_study_spend_usd,
        projection.budget_usd,
        "OVER THE CAP" if projection.over_cap else "within the cap",
    )
    if projection.unpriced_pipelines:
        logger.warning(
            "Unpriced (excluded from the USD projection, shown as call counts): {}",
            projection.unpriced_pipelines,
        )
    if projection.over_cap:
        logger.error("Projected spend exceeds the cap: STOP and ask the user before any paid call.")
        return EXIT_OVER_CAP
    return EXIT_OK


# -------------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select-frames", help="write or verify conf/cost_val_frames.json")
    sel.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    sel.add_argument("--check", action="store_true")
    sel.set_defaults(func=cmd_select_frames)

    chk = sub.add_parser("check-config", help="resolve all pipelines offline")
    chk.set_defaults(func=cmd_check_config)

    run = sub.add_parser("run", help="one pipeline x one run (paid unless --dry-run)")
    run.add_argument("--pipeline", required=True)
    run.add_argument("--run", required=True, choices=["1", "2", "proj"])
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    run.set_defaults(func=cmd_run)

    proj = sub.add_parser("project", help="project the study spend from the proj runs")
    proj.set_defaults(func=cmd_project)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
