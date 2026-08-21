"""Loader + renderer for the VLM ablation log.

The ablation table makes a claim no other report table makes: that a change was
*kept*. That verdict is derived by comparing each element's val winner against
the published ``vlm_zeroshot.yaml``, so the interesting failures are the ones
where the table would say something the config does not support — claiming a
win was adopted when it was not, or dressing up a rounding-error delta as a
result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from object_detection_eval.report import (
    ABLATION_NOISE_FLOOR,
    ReportLoadError,
    ablation_headline_table,
    ablation_summary_table,
    load_ablation_log,
    load_zeroshot_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_LOG = _REPO_ROOT / "benchmarks" / "basketball" / "results" / "vlm" / "ablation"
_COMMITTED_CONF = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "vlm_zeroshot.yaml"


def _arm(
    arm: str,
    model: str,
    element: str,
    map_50_95: float,
    config: dict[str, Any],
    baseline: str | None = None,
    delta: float | None = None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "model": model,
        "element": element,
        "baseline": baseline,
        "mAP_50_95": map_50_95,
        "mAP_50": map_50_95 * 1.5,
        "per_class_ap50": {"player": map_50_95},
        "delta_map5095": delta,
        "config": config,
    }


def _write_log(path: Path, arms: list[dict[str, Any]], split: str = "valid") -> Path:
    path.write_text(json.dumps({"split": split, "arms": arms}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_loads_the_committed_log_if_present() -> None:
    log_path = _COMMITTED_LOG / "valid_arms.json"
    if not log_path.is_file():
        pytest.skip("ablation log not committed yet")
    log = load_ablation_log(log_path)
    assert log.split == "valid"
    assert log.arms


def test_a_log_scored_on_the_published_split_is_rejected(tmp_path: Path) -> None:
    """A protocol failure, caught at load rather than rendered.

    An ablation is a search. If its log says it ran on test, the configuration
    was chosen on the split the report publishes and every number downstream is
    the maximum over the arms tried — not something to render a table from.
    """
    path = _write_log(
        tmp_path / "test_arms.json",
        [_arm("a", "owlv2", "baseline", 0.2, {"nms_iou_threshold": 0.3})],
        split="test",
    )
    with pytest.raises(ReportLoadError, match="split='test'"):
        load_ablation_log(path)


def test_unknown_key_fails_at_load(tmp_path: Path) -> None:
    bad = _arm("a", "owlv2", "baseline", 0.2, {})
    bad["surprise"] = 1
    path = _write_log(tmp_path / "valid_arms.json", [bad])
    with pytest.raises(ReportLoadError):
        load_ablation_log(path)


def test_zeroshot_config_strips_bookkeeping_keys() -> None:
    config = load_zeroshot_config(_COMMITTED_CONF)
    assert "owlv2" in config
    assert "name" not in config["owlv2"]
    assert "expected_map5095" not in config["owlv2"]
    assert "classes" in config["owlv2"]


def test_zeroshot_config_rejects_a_file_without_models(tmp_path: Path) -> None:
    path = tmp_path / "conf.yaml"
    path.write_text(yaml.safe_dump({"tolerance": 0.02}), encoding="utf-8")
    with pytest.raises(ReportLoadError, match="`models` list"):
        load_zeroshot_config(path)


# ---------------------------------------------------------------------------
# The kept/reverted verdict
# ---------------------------------------------------------------------------


def _log_with_winner(tmp_path: Path, winner_value: float, delta: float) -> Any:
    return load_ablation_log(
        _write_log(
            tmp_path / "valid_arms.json",
            [
                _arm("owlv2__baseline", "owlv2", "baseline", 0.240, {"nms_iou_threshold": 0.3}),
                _arm(
                    "owlv2__nms_iou__x",
                    "owlv2",
                    "nms_iou",
                    0.240 + delta,
                    {"nms_iou_threshold": winner_value},
                    baseline="owlv2__baseline",
                    delta=delta,
                ),
            ],
        )
    )


def test_a_win_the_config_adopted_is_marked_kept(tmp_path: Path) -> None:
    log = _log_with_winner(tmp_path, winner_value=0.9, delta=0.0075)
    table = ablation_summary_table(log, {"owlv2": {"nms_iou_threshold": 0.9}})
    assert "**kept**" in table


def test_a_win_the_config_did_not_adopt_is_not_marked_kept(tmp_path: Path) -> None:
    """The table describes the shipped configuration, not the best arm found.

    Without this, adopting nothing and rendering the log would still produce a
    table full of "kept" — a claim about the published config that the published
    config does not support.
    """
    log = _log_with_winner(tmp_path, winner_value=0.9, delta=0.0075)
    table = ablation_summary_table(log, {"owlv2": {"nms_iou_threshold": 0.3}})
    assert "**kept**" not in table
    assert "reverted" in table


def test_a_delta_inside_the_noise_floor_is_reverted_even_if_adopted(tmp_path: Path) -> None:
    """96 val images do not resolve a thousandth of a point.

    Calling a +0.0005 difference a win is fitting the val split, which is the
    same error as fitting the test split one step removed.
    """
    log = _log_with_winner(tmp_path, winner_value=0.9, delta=ABLATION_NOISE_FLOOR / 4)
    table = ablation_summary_table(log, {"owlv2": {"nms_iou_threshold": 0.9}})
    assert "within noise" in table
    assert "**kept**" not in table


def test_the_best_arm_is_the_one_reported(tmp_path: Path) -> None:
    log = load_ablation_log(
        _write_log(
            tmp_path / "valid_arms.json",
            [
                _arm("owlv2__baseline", "owlv2", "baseline", 0.240, {"nms_iou_threshold": 0.3}),
                _arm(
                    "owlv2__nms_iou__0.5",
                    "owlv2",
                    "nms_iou",
                    0.245,
                    {"nms_iou_threshold": 0.5},
                    baseline="owlv2__baseline",
                    delta=0.005,
                ),
                _arm(
                    "owlv2__nms_iou__0.9",
                    "owlv2",
                    "nms_iou",
                    0.248,
                    {"nms_iou_threshold": 0.9},
                    baseline="owlv2__baseline",
                    delta=0.008,
                ),
            ],
        )
    )
    table = ablation_summary_table(log, {"owlv2": {"nms_iou_threshold": 0.9}})
    assert "`0.9`" in table
    assert "0.248" in table
    # Both arms counted, so the reader can see the element was actually swept.
    assert "| 2 |" in table


def test_an_arm_without_a_comparable_baseline_says_so(tmp_path: Path) -> None:
    """Better an honest gap than a delta against a number from another run."""
    log = load_ablation_log(
        _write_log(
            tmp_path / "valid_arms.json",
            [
                _arm("owlv2__baseline", "owlv2", "baseline", 0.240, {"nms_iou_threshold": 0.3}),
                _arm(
                    "owlv2__nms_iou__0.9",
                    "owlv2",
                    "nms_iou",
                    0.248,
                    {"nms_iou_threshold": 0.9},
                    baseline="owlv2__baseline",
                    delta=None,
                ),
            ],
        )
    )
    assert "not comparable" in ablation_summary_table(log, {"owlv2": {}})


def test_baseline_arms_do_not_get_their_own_row(tmp_path: Path) -> None:
    log = _log_with_winner(tmp_path, winner_value=0.9, delta=0.0075)
    table = ablation_summary_table(log, {"owlv2": {"nms_iou_threshold": 0.9}})
    assert "baseline" not in table
    assert len([line for line in table.splitlines() if line.startswith("| owlv2")]) == 1


# ---------------------------------------------------------------------------
# The headline table -- same "describes what shipped" discipline as the
# per-element table above, but for the one-row-per-model summary.
# ---------------------------------------------------------------------------


def test_headline_reports_the_published_arm_not_the_best_ever_logged(tmp_path: Path) -> None:
    """A disclosed-but-not-adopted arm must not read as the outcome.

    A targeted follow-up can log an arm that outscores what actually shipped
    (e.g. it improves a class unrelated to the one it was investigating, and
    that improvement is disclosed but not adopted). The headline table's job is
    to report what the manifest actually runs, not the single highest-scoring
    arm the log happens to contain -- otherwise it would attribute a change
    that was explicitly not kept to the model, exactly the failure mode
    `ablation_summary_table` already guards against.
    """
    log = load_ablation_log(
        _write_log(
            tmp_path / "valid_arms.json",
            [
                _arm(
                    "yolo_world__baseline",
                    "yolo_world",
                    "baseline",
                    0.132,
                    {"nms_iou_threshold": 0.5, "classes": ["referee"]},
                ),
                _arm(
                    "yolo_world__published",
                    "yolo_world",
                    "referee_word",
                    0.177,
                    {"nms_iou_threshold": 0.7, "classes": ["referee"]},
                    baseline="yolo_world__baseline",
                    delta=0.045,
                ),
                _arm(
                    "yolo_world__not_adopted",
                    "yolo_world",
                    "referee_word",
                    0.185,
                    {"nms_iou_threshold": 0.7, "classes": ["official"]},
                    baseline="yolo_world__baseline",
                    delta=0.053,
                ),
            ],
        )
    )
    table = ablation_headline_table(
        log, {"yolo_world": {"nms_iou_threshold": 0.7, "classes": ["referee"]}}
    )
    assert "0.177" in table
    assert "0.185" not in table


def test_headline_says_none_adopted_when_nothing_published_beats_baseline(
    tmp_path: Path,
) -> None:
    """A model can log a higher-scoring arm and still have adopted nothing.

    Without restricting the candidate pool to arms matching the published
    config, this case would report the unadopted winner as if it were kept.
    """
    log = load_ablation_log(
        _write_log(
            tmp_path / "valid_arms.json",
            [
                _arm("gemini__baseline", "gemini", "baseline", 0.258, {"nms_iou_threshold": None}),
                _arm(
                    "gemini__not_adopted",
                    "gemini",
                    "tiles",
                    0.290,
                    {"nms_iou_threshold": 0.5},
                    baseline="gemini__baseline",
                    delta=0.032,
                ),
            ],
        )
    )
    table = ablation_headline_table(log, {"gemini": {"nms_iou_threshold": None}})
    assert "none — baseline beat every arm tried" in table
    assert "0.290" not in table


def test_headline_ignores_published_fields_the_ablation_schema_does_not_track(
    tmp_path: Path,
) -> None:
    """A manifest-only field (e.g. Florence-2's derived `caption`) must not
    block a match. It has no independent Arm knob, so requiring it to equal
    the manifest value would make every arm look unpublished."""
    log = load_ablation_log(
        _write_log(
            tmp_path / "valid_arms.json",
            [
                _arm("florence2__baseline", "florence2", "baseline", 0.125, {"tiles": None}),
                _arm(
                    "florence2__published",
                    "florence2",
                    "combined",
                    0.234,
                    {"tiles": [2, 2]},
                    baseline="florence2__baseline",
                    delta=0.109,
                ),
            ],
        )
    )
    published = {
        "florence2": {
            "tiles": [2, 2],
            # Not a real Arm field -- derived from `classes` at run time.
            "caption": "player. ball. referee. rim. number.",
        }
    }
    table = ablation_headline_table(log, published)
    assert "0.234" in table
    assert "none — baseline beat every arm tried" not in table
