"""Loader + renderers for the VLM fusion sweep.

The fusion section makes two claims the rest of the report does not: that a
measured gain belongs to a specific *mechanism*, and that mAP and label quality
rank the configurations differently. Both are structural properties of how the
rows are picked, so the failures worth pinning are the ones where the table
would still render but stop meaning what the prose says it means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from object_detection_eval.report import (
    ReportLoadError,
    fusion_headline_table,
    fusion_label_quality_table,
    fusion_subset_table,
    fusion_test_table,
    load_fusion_log,
    load_fusion_test_log,
    load_vlm_metrics,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FUSION_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "results" / "vlm" / "fusion"
_COMMITTED = _FUSION_DIR / "valid_fusion.json"
_COMMITTED_TEST = _FUSION_DIR / "test_fusion.json"


def _row(
    models: list[str],
    method: str,
    map_50_95: float,
    *,
    iou: float = 0.55,
    normalize: bool = False,
    min_models: int | None = None,
    boxes: float = 100.0,
    recall_at_p95: float | None = 0.4,
) -> dict[str, Any]:
    return {
        "models": models,
        "n_models": len(models),
        "method": method,
        "iou": iou,
        "normalize": normalize,
        "min_models": min_models,
        "mAP_50_95": map_50_95,
        "mAP_50": map_50_95 + 0.1,
        "per_class_ap50": {"player": 0.8},
        "boxes_per_image": boxes,
        "best_f1": {"threshold": 0.3, "precision": 0.8, "recall": 0.6, "f1": 0.69},
        "recall_at_p90": None,
        "recall_at_p95": (
            None
            if recall_at_p95 is None
            else {"threshold": 0.5, "precision": 0.95, "recall": recall_at_p95}
        ),
    }


def _log(rows: list[dict[str, Any]], split: str = "valid") -> dict[str, Any]:
    return {
        "split": split,
        "default_iou": 0.55,
        "adopted_arms": {"owlv2": "owlv2__nms_on_tiles__0.5"},
        "rows": rows,
    }


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "fusion.json"
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_fusion_log_rejects_a_test_split(tmp_path: Path) -> None:
    """The guard the whole protocol rests on.

    A sweep this wide scored on `test` reports its own argmax. That is a
    protocol failure, not a rendering problem, so it fails at load rather than
    rendering a plausible-looking table.
    """
    path = _write(tmp_path, _log([_row(["owlv2"], "nms", 0.3)], split="test"))
    with pytest.raises(ReportLoadError, match="test"):
        load_fusion_log(path)


def test_load_fusion_log_rejects_unknown_keys(tmp_path: Path) -> None:
    payload = _log([_row(["owlv2"], "nms", 0.3)])
    payload["rows"][0]["surprise"] = 1
    with pytest.raises(ReportLoadError):
        load_fusion_log(_write(tmp_path, payload))


def test_load_fusion_log_accepts_a_missing_operating_point(tmp_path: Path) -> None:
    """A model that never reaches 95% precision is a result, not a defect.

    Florence-2 and Gemini pin every confidence at 1.0, so they have exactly one
    operating point and no threshold reaches the bar. The schema must carry that
    as `None` rather than refusing to load.
    """
    path = _write(tmp_path, _log([_row(["gemini"], "nms", 0.25, recall_at_p95=None)]))
    log = load_fusion_log(path)
    assert log.rows[0].recall_at_p95 is None


# ---------------------------------------------------------------------------
# Headline table
# ---------------------------------------------------------------------------


def _headline_log() -> Any:
    return _log(
        [
            _row(["owlv2"], "nms", 0.2879, iou=1.0, boxes=510),
            _row(["gemini"], "nms", 0.2583, iou=1.0, boxes=17),
            _row(list("abcdef"), "nms", 0.2904, boxes=1469),
            _row(list("abcdef"), "agree", 0.3500, boxes=1469),
            _row(list("abcdef"), "wbf", 0.4085, boxes=1469),
        ]
    )


def test_headline_baseline_is_the_best_single_model(tmp_path: Path) -> None:
    table = fusion_headline_table(load_fusion_log(_write(tmp_path, _headline_log())))
    assert "Best single model (owlv2)" in table
    assert "0.288" in table


def test_headline_deltas_are_all_against_that_baseline(tmp_path: Path) -> None:
    """Each step's delta must be against the single model, not the row above it.

    Reporting step-over-step deltas would make the column look like an
    attribution while summing to the wrong total -- the exact failure mode the
    stacking rule in the ablation exists to prevent.
    """
    table = fusion_headline_table(load_fusion_log(_write(tmp_path, _headline_log())))
    assert "+0.0025" in table  # nms   - owlv2
    assert "+0.0621" in table  # agree - owlv2
    assert "+0.1206" in table  # wbf   - owlv2


def test_headline_ignores_rank_normalised_rows(tmp_path: Path) -> None:
    """The headline reports the adopted variant; the control lives in the log.

    Rank normalisation lost, and a table that silently mixed the two would
    report a number no single configuration produced.
    """
    payload = _headline_log()
    payload["rows"].append(_row(list("abcdef"), "wbf", 0.9999, normalize=True))
    table = fusion_headline_table(load_fusion_log(_write(tmp_path, payload)))
    assert "0.9999" not in table
    assert "1.000" not in table


def test_headline_ignores_off_default_iou(tmp_path: Path) -> None:
    """Sensitivity values are reported elsewhere and never adopted."""
    payload = _headline_log()
    payload["rows"].append(_row(list("abcdef"), "wbf", 0.8888, iou=0.7))
    table = fusion_headline_table(load_fusion_log(_write(tmp_path, payload)))
    assert "0.889" not in table


# ---------------------------------------------------------------------------
# Label-quality table
# ---------------------------------------------------------------------------


def test_label_quality_reports_no_operating_point_explicitly(tmp_path: Path) -> None:
    """A blank cell would read as a missing measurement rather than a finding."""
    payload = _log(
        [
            _row(["gemini"], "nms", 0.2583, iou=1.0, recall_at_p95=None),
            _row(["a", "b"], "wbf", 0.40, recall_at_p95=0.55),
        ]
    )
    table = fusion_label_quality_table(load_fusion_log(_write(tmp_path, payload)))
    assert "never reaches 95%" in table


def test_label_quality_orders_by_map_so_the_disagreement_is_visible(tmp_path: Path) -> None:
    """The table's whole argument is that the two rankings disagree.

    Sorting single models by mAP while showing recall@P95 beside it is what
    makes "the mAP winner is the worst labeler" legible at a glance; sorting by
    the label metric would hide it.
    """
    payload = _log(
        [
            _row(["owlv2"], "nms", 0.2879, iou=1.0, recall_at_p95=0.010),
            _row(["grounding_dino"], "nms", 0.2779, iou=1.0, recall_at_p95=0.168),
            _row(["a", "b"], "wbf", 0.4085, recall_at_p95=0.552),
        ]
    )
    table = fusion_label_quality_table(load_fusion_log(_write(tmp_path, payload)))
    lines = [ln for ln in table.splitlines() if ln.startswith("| ")]
    assert "owlv2" in lines[2]
    assert "grounding_dino" in lines[3]


# ---------------------------------------------------------------------------
# Subset table
# ---------------------------------------------------------------------------


def test_subset_table_reports_the_best_at_each_size(tmp_path: Path) -> None:
    payload = _log(
        [
            _row(["owlv2"], "nms", 0.2879, iou=1.0),
            _row(["a", "b"], "wbf", 0.30),
            _row(["a", "c"], "wbf", 0.34),
            _row(["a", "b", "c"], "wbf", 0.39),
        ]
    )
    table = fusion_subset_table(load_fusion_log(_write(tmp_path, payload)))
    assert "a, c" in table
    assert "a, b, c" in table
    assert "0.300" not in table  # the losing 2-model subset is not the row shown


def test_subset_table_excludes_non_wbf_and_normalised_rows(tmp_path: Path) -> None:
    """Mixing operators would make the size curve compare different things."""
    payload = _log(
        [
            _row(["owlv2"], "nms", 0.2879, iou=1.0),
            _row(["a", "b"], "wbf", 0.30),
            _row(["a", "b"], "consensus", 0.77, min_models=2),
            _row(["a", "b"], "wbf", 0.88, normalize=True),
        ]
    )
    table = fusion_subset_table(load_fusion_log(_write(tmp_path, payload)))
    assert "0.770" not in table
    assert "0.880" not in table
    assert "0.300" in table


# ---------------------------------------------------------------------------
# The committed log
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _COMMITTED.exists(), reason="fusion sweep not yet committed")
def test_committed_log_loads_and_is_val_only() -> None:
    log = load_fusion_log(_COMMITTED)
    assert log.split == "valid"
    assert log.rows
    assert any(r.n_models == 1 for r in log.rows), "log must carry its own baselines"


# ---------------------------------------------------------------------------
# The final test scoring
# ---------------------------------------------------------------------------


def test_load_fusion_test_log_requires_the_test_split(tmp_path: Path) -> None:
    payload = _log([_row(["owlv2"], "nms", 0.31, iou=1.0), _row(["a", "b"], "wbf", 0.40)])
    with pytest.raises(ReportLoadError, match="test"):
        load_fusion_test_log(_write(tmp_path, payload))


def test_load_fusion_test_log_rejects_multiple_ensembles(tmp_path: Path) -> None:
    """More than one fused row means test was scored against several candidates.

    That is the definition of turning the published split into a second
    validation split, and it must fail at load rather than render a table whose
    best row a reader would assume had been chosen honestly.
    """
    payload = _log(
        [
            _row(["owlv2"], "nms", 0.31, iou=1.0),
            _row(["a", "b"], "wbf", 0.40),
            _row(["a", "b", "c"], "wbf", 0.42),
        ],
        split="test",
    )
    with pytest.raises(ReportLoadError, match="scored once"):
        load_fusion_test_log(_write(tmp_path, payload))


def test_fusion_test_table_shows_the_delta_over_the_best_single(tmp_path: Path) -> None:
    payload = _log(
        [
            _row(["owlv2"], "nms", 0.3148, iou=1.0),
            _row(["gemini"], "nms", 0.2497, iou=1.0),
            _row(list("abcdef"), "wbf", 0.4061),
        ],
        split="test",
    )
    table = fusion_test_table(load_fusion_test_log(_write(tmp_path, payload)))
    assert "+0.0913" in table
    assert "All 6 fused" in table


@pytest.mark.skipif(not _COMMITTED_TEST.exists(), reason="test scoring not yet committed")
def test_committed_test_log_matches_the_published_per_model_numbers() -> None:
    """The fusion plumbing must not have altered detections on their way past.

    Both this file and `vlm_metrics_merged5.json` are the same committed dumps
    through the same scorer, so any divergence means one of the two tables in
    the report is describing a pipeline the other does not use -- which is
    exactly how PR #17 shipped a wrong number.

    The two files are NOT expected to cover the same model SET, only to agree
    on whichever models they share: `vlm_metrics_merged5.json` holds every
    published VLM row, while this log holds only the models `adopted_arms`
    names as fused (six, as of LLMDet-large's addition -- it is not yet part
    of fusion, a documented follow-up, so it appears in the former but not
    the latter).
    """
    published = load_vlm_metrics(
        _REPO_ROOT / "benchmarks" / "basketball" / "results" / "vlm" / "vlm_metrics_merged5.json"
    )

    # The two files key models differently -- display names ("Grounding-DINO")
    # against manifest names ("grounding_dino") -- so match on alphanumerics.
    def key(name: str) -> str:
        return "".join(c for c in name.lower() if c.isalnum())

    by_key = {key(k): v for k, v in published.items()}

    log = load_fusion_test_log(_COMMITTED_TEST)
    checked = 0
    for row in log.rows:
        if row.n_models != 1:
            continue
        entry = by_key.get(key(row.models[0]))
        assert entry is not None, f"no published metrics for {row.models[0]}"
        assert row.map_50_95 == pytest.approx(entry["mAP_50_95"], abs=1e-6)
        checked += 1
    assert checked == len(log.adopted_arms), "every fused model must appear in the test scoring"
