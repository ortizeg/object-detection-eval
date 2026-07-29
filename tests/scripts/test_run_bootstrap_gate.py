"""Tests for scripts/run_bootstrap_gate.py (REPRO-03) -- offline, dataset-free.

Locks two things: (1) the same-seed determinism the gate relies on to
reproduce the anchor CIs (run_bootstrap twice at seed=0 gives byte-identical
per-model and pairwise-difference CI arrays), and (2) the pure tie/
significance and anchor-tolerance helpers the gate uses to grade its two
checks. The script lives outside `src/` (a CLI entry point, not library
code), so it is loaded here via `importlib` from its file path, mirroring
`tests/scripts/test_run_benchmark.py`. This never reads the source-repo
predictions, the published anchor json, or the basketball dataset that
run_bootstrap_gate needs at runtime, so the whole suite stays green offline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import supervision as sv

from object_detection_eval.metrics.bootstrap import build_report, run_bootstrap

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_bootstrap_gate.py"


def _load_run_bootstrap_gate_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_bootstrap_gate", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module: pydantic's `from __future__
    # import annotations` string-annotation resolution looks the module up
    # via sys.modules[model.__module__], which is empty until this happens.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_bootstrap_gate = _load_run_bootstrap_gate_module()


def _det(boxes: list[list[float]], class_ids: list[int], confs: list[float]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confs, dtype=np.float32),
    )


def _synthetic_gt_map() -> dict[str, sv.Detections]:
    """A small synthetic ground truth over 6 images, single class."""
    return {
        f"img{i}.jpg": sv.Detections(
            xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
            class_id=np.array([0]),
        )
        for i in range(6)
    }


def _synthetic_pred_maps(
    gt_map: dict[str, sv.Detections],
) -> dict[str, dict[str, sv.Detections]]:
    """Two synthetic models: one near-perfect, one noisier (weaker recall)."""
    model_a: dict[str, sv.Detections] = {}
    model_b: dict[str, sv.Detections] = {}
    for i, filename in enumerate(gt_map):
        model_a[filename] = _det([[100, 100, 300, 400]], [0], [0.9])
        if i % 2 == 0:
            model_b[filename] = _det([[100, 100, 300, 400]], [0], [0.8])
        else:
            model_b[filename] = sv.Detections.empty()
    return {"model_a": model_a, "model_b": model_b}


class TestBootstrapDeterminismLocksTheGate:
    """Same-seed determinism the gate relies on to reproduce the anchor CIs."""

    def test_same_seed_yields_identical_per_model_and_pairwise_ci(self) -> None:
        gt_map = _synthetic_gt_map()
        pred_maps = _synthetic_pred_maps(gt_map)

        boot_1 = run_bootstrap(gt_map, pred_maps, n_boot=30, seed=0)
        boot_2 = run_bootstrap(gt_map, pred_maps, n_boot=30, seed=0)
        report_1 = build_report(gt_map, pred_maps, boot_1, n_boot=30, seed=0)
        report_2 = build_report(gt_map, pred_maps, boot_2, n_boot=30, seed=0)

        for model in pred_maps:
            stats_1 = report_1["per_model"][model]["mAP_50_95"]
            stats_2 = report_2["per_model"][model]["mAP_50_95"]
            assert stats_1["ci_2.5"] == stats_2["ci_2.5"]
            assert stats_1["ci_97.5"] == stats_2["ci_97.5"]

        pair_1 = report_1["pairwise"]["model_a minus model_b"]["mAP_50_95"]
        pair_2 = report_2["pairwise"]["model_a minus model_b"]["mAP_50_95"]
        assert pair_1["ci_2.5"] == pair_2["ci_2.5"]
        assert pair_1["ci_97.5"] == pair_2["ci_97.5"]
        assert pair_1["point_diff"] == pair_2["point_diff"]


class TestWriteBootstrapResults:
    """The --write-results helper round-trips build_report's dict on disk."""

    @staticmethod
    def _minimal_report() -> dict:
        """A minimal build_report()-shaped dict (plain floats/bools)."""
        return {
            "config": {"n_boot": 1000, "seed": 0, "n_images": 94, "models": ["A", "B"]},
            "per_model": {
                "A": {"mAP_50_95": {"point_estimate": 0.716, "ci_2.5": 0.704, "ci_97.5": 0.729}},
                "B": {"mAP_50_95": {"point_estimate": 0.628, "ci_2.5": 0.615, "ci_97.5": 0.641}},
            },
            "pairwise": {
                "A minus B": {
                    "mAP_50_95": {
                        "point_diff": 0.088,
                        "ci_2.5": 0.070,
                        "ci_97.5": 0.106,
                        "ci_excludes_zero": True,
                    }
                },
            },
        }

    def test_round_trip_preserves_per_model_and_pairwise(self, tmp_path: Path) -> None:
        report = self._minimal_report()
        out = tmp_path / "nested" / "bootstrap.json"

        run_bootstrap_gate.write_bootstrap_results(out, report)

        assert out.is_file()  # parent dirs created
        reloaded = json.loads(out.read_text())
        assert reloaded == report
        assert reloaded["per_model"]["A"]["mAP_50_95"]["ci_2.5"] == 0.704
        assert reloaded["pairwise"]["A minus B"]["mAP_50_95"]["point_diff"] == 0.088

    def test_round_trip_preserves_ci_excludes_zero_bool(self, tmp_path: Path) -> None:
        # A downstream reader must derive significance from this bool, never a
        # hand-typed "all significant" sentence -- so it must survive as a bool.
        report = self._minimal_report()
        report["pairwise"]["A minus B"]["mAP_50_95"]["ci_excludes_zero"] = False
        out = tmp_path / "bootstrap.json"

        run_bootstrap_gate.write_bootstrap_results(out, report)

        reloaded = json.loads(out.read_text())
        value = reloaded["pairwise"]["A minus B"]["mAP_50_95"]["ci_excludes_zero"]
        assert value is False
        assert isinstance(value, bool)


class TestTieClassifier:
    """The tie-classifier reports a tie iff the CI straddles zero."""

    def test_straddling_ci_is_a_tie(self) -> None:
        assert run_bootstrap_gate.is_tie(-0.0033, 0.0190) is True
        assert run_bootstrap_gate.ci_excludes_zero(-0.0033, 0.0190) is False

    def test_ci_excluding_zero_above_is_significant_not_a_tie(self) -> None:
        assert run_bootstrap_gate.ci_excludes_zero(0.0141, 0.0446) is True
        assert run_bootstrap_gate.is_tie(0.0141, 0.0446) is False

    def test_ci_excluding_zero_below_is_significant_not_a_tie(self) -> None:
        assert run_bootstrap_gate.ci_excludes_zero(-0.0446, -0.0141) is True
        assert run_bootstrap_gate.is_tie(-0.0446, -0.0141) is False

    def test_boundary_touching_zero_is_not_excluded(self) -> None:
        # lower == 0.0 -> `lower > 0.0` is False and `upper < 0.0` is False,
        # so the CI does not exclude zero at the exact boundary.
        assert run_bootstrap_gate.ci_excludes_zero(0.0, 0.02) is False
        assert run_bootstrap_gate.is_tie(0.0, 0.02) is True


class TestAnchorMatches:
    """The anchor-comparison helper passes within tolerance, fails outside it."""

    def test_exact_match_passes(self) -> None:
        anchor = {"point_estimate": 0.7155, "ci_2.5": 0.7041, "ci_97.5": 0.7285}
        measured = dict(anchor)
        assert run_bootstrap_gate.anchor_matches(measured, anchor, tolerance=0.01) is True

    def test_within_tolerance_passes(self) -> None:
        anchor = {"point_estimate": 0.7155, "ci_2.5": 0.7041, "ci_97.5": 0.7285}
        measured = {"point_estimate": 0.7160, "ci_2.5": 0.7045, "ci_97.5": 0.7290}
        assert run_bootstrap_gate.anchor_matches(measured, anchor, tolerance=0.01) is True

    def test_out_of_tolerance_ci_bound_fails(self) -> None:
        anchor = {"point_estimate": 0.7155, "ci_2.5": 0.7041, "ci_97.5": 0.7285}
        # ci_97.5 is 0.02 off -- outside a 0.01 tolerance.
        measured = {"point_estimate": 0.7155, "ci_2.5": 0.7041, "ci_97.5": 0.7485}
        assert run_bootstrap_gate.anchor_matches(measured, anchor, tolerance=0.01) is False

    def test_out_of_tolerance_point_estimate_fails(self) -> None:
        anchor = {"point_estimate": 0.7155, "ci_2.5": 0.7041, "ci_97.5": 0.7285}
        measured = {"point_estimate": 0.7355, "ci_2.5": 0.7041, "ci_97.5": 0.7285}
        assert run_bootstrap_gate.anchor_matches(measured, anchor, tolerance=0.01) is False


class TestWithinTolerance:
    """The pure tolerance helper the gate reuses for the tie-comparison too."""

    def test_exact_match_at_boundary_passes(self) -> None:
        measured, expected = 0.714, 0.716
        tolerance = abs(measured - expected)
        assert run_bootstrap_gate.within_tolerance(measured, expected, tolerance) is True

    def test_just_outside_fails(self) -> None:
        assert run_bootstrap_gate.within_tolerance(0.700, 0.716, 0.01) is False


class TestAdjacentPairReproducesAnchorSignificance:
    """Check A grades adjacent pairs by REPRODUCING the anchor's recorded
    ``ci_excludes_zero`` per pair -- not a blanket 'all significant'.

    The anchor records RTMDet-M vs DAMO-YOLO-M (a 0.85pt gap) as a statistical
    tie (``ci_excludes_zero`` False); the original report prose ('every adjacent
    pair is significant') over-claimed. Faithful reproduction must classify that
    pair as a tie too, and the gate must treat that as a PASS.
    """

    def test_rtmdet_vs_damo_ci_classifies_as_a_tie(self) -> None:
        # The anchor's RTMDet-M minus DAMO-YOLO-M CI (mAP_50:95), which our
        # bootstrap reproduced to 4 decimals -> straddles zero -> a tie.
        assert run_bootstrap_gate.ci_excludes_zero(-0.0022, 0.0200) is False

    def test_pair_reproduces_when_measured_significance_matches_anchor(self) -> None:
        # A tie pair reproduces the anchor's recorded tie (False == False).
        measured_tie = run_bootstrap_gate.ci_excludes_zero(-0.0022, 0.0200)
        assert (measured_tie == False) is True  # noqa: E712 - explicit vs anchor value
        # A significant pair reproduces the anchor's recorded significance (True).
        measured_sig = run_bootstrap_gate.ci_excludes_zero(0.0141, 0.0436)
        assert (measured_sig == True) is True  # noqa: E712 - explicit vs anchor value
