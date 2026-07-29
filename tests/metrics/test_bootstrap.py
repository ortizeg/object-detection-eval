"""Tests for object_detection_eval.metrics.bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from object_detection_eval.metrics.bootstrap import (
    build_report,
    load_predictions,
    percentile_ci,
    resample_map,
    run_bootstrap,
)


def _det(boxes: list[list[float]], class_ids: list[int], confs: list[float]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confs, dtype=np.float32),
    )


@pytest.fixture
def gt_map() -> dict[str, sv.Detections]:
    """A small synthetic ground truth over 6 images, single class."""
    return {
        f"img{i}.jpg": sv.Detections(
            xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
            class_id=np.array([0]),
        )
        for i in range(6)
    }


@pytest.fixture
def pred_maps(gt_map: dict[str, sv.Detections]) -> dict[str, dict[str, sv.Detections]]:
    """Two synthetic models: one near-perfect, one noisier."""
    model_a: dict[str, sv.Detections] = {}
    model_b: dict[str, sv.Detections] = {}
    for i, filename in enumerate(gt_map):
        # model_a: perfect match every image.
        model_a[filename] = _det([[100, 100, 300, 400]], [0], [0.9])
        # model_b: perfect on even images, empty on odd (weaker recall).
        if i % 2 == 0:
            model_b[filename] = _det([[100, 100, 300, 400]], [0], [0.8])
        else:
            model_b[filename] = sv.Detections.empty()
    return {"model_a": model_a, "model_b": model_b}


class TestLoadPredictions:
    """Tests for load_predictions."""

    def test_loads_dets_from_json(self, tmp_path: Path) -> None:
        raw = {
            "img.jpg": [{"bbox_xyxy": [1.0, 2.0, 3.0, 4.0], "class_id": 0, "confidence": 0.9}],
            "empty.jpg": [],
        }
        path = tmp_path / "preds.json"
        path.write_text(json.dumps(raw))

        result = load_predictions(path)

        assert len(result["img.jpg"]) == 1
        assert result["img.jpg"].class_id is not None
        assert result["img.jpg"].class_id[0] == 0
        assert len(result["empty.jpg"]) == 0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_predictions(tmp_path / "does_not_exist.json")


class TestResampleMap:
    """Tests for resample_map."""

    def test_positional_keys_allow_duplicates(self) -> None:
        source = {"a.jpg": _det([[0, 0, 1, 1]], [0], [0.5])}
        filenames = ["a.jpg"]
        # Same image drawn 3 times.
        draw = np.array([0, 0, 0])

        resampled = resample_map(source, filenames, draw)

        assert set(resampled.keys()) == {"a.jpg__0", "a.jpg__1", "a.jpg__2"}
        assert len(resampled) == 3

    def test_missing_prediction_falls_back_to_empty(self) -> None:
        source: dict[str, sv.Detections] = {}
        filenames = ["a.jpg"]
        draw = np.array([0])

        resampled = resample_map(source, filenames, draw)

        assert len(resampled["a.jpg__0"]) == 0


class TestPercentileCi:
    """Tests for percentile_ci."""

    def test_bounds_ordering(self) -> None:
        values = np.linspace(0.0, 1.0, 1000)
        lower, upper = percentile_ci(values)
        assert lower < upper
        assert lower == pytest.approx(0.025, abs=0.01)
        assert upper == pytest.approx(0.975, abs=0.01)


class TestRunBootstrapDeterminism:
    """Determinism tests for run_bootstrap + build_report (CORE-04)."""

    def test_same_seed_same_single_model_ci(
        self,
        gt_map: dict[str, sv.Detections],
        pred_maps: dict[str, dict[str, sv.Detections]],
    ) -> None:
        boot_1 = run_bootstrap(gt_map, pred_maps, n_boot=50, seed=0)
        boot_2 = run_bootstrap(gt_map, pred_maps, n_boot=50, seed=0)

        report_1 = build_report(gt_map, pred_maps, boot_1, n_boot=50, seed=0)
        report_2 = build_report(gt_map, pred_maps, boot_2, n_boot=50, seed=0)

        for model in pred_maps:
            for metric in ("mAP_50_95", "mAP_50"):
                stats_1 = report_1["per_model"][model][metric]
                stats_2 = report_2["per_model"][model][metric]
                assert stats_1["ci_2.5"] == stats_2["ci_2.5"]
                assert stats_1["ci_97.5"] == stats_2["ci_97.5"]
                assert stats_1["bootstrap_mean"] == stats_2["bootstrap_mean"]

    def test_same_seed_same_pairwise_diff_ci(
        self,
        gt_map: dict[str, sv.Detections],
        pred_maps: dict[str, dict[str, sv.Detections]],
    ) -> None:
        boot_1 = run_bootstrap(gt_map, pred_maps, n_boot=50, seed=0)
        boot_2 = run_bootstrap(gt_map, pred_maps, n_boot=50, seed=0)

        report_1 = build_report(gt_map, pred_maps, boot_1, n_boot=50, seed=0)
        report_2 = build_report(gt_map, pred_maps, boot_2, n_boot=50, seed=0)

        pair_key = "model_a minus model_b"
        for metric in ("mAP_50_95", "mAP_50"):
            diff_1 = report_1["pairwise"][pair_key][metric]
            diff_2 = report_2["pairwise"][pair_key][metric]
            assert diff_1["ci_2.5"] == diff_2["ci_2.5"]
            assert diff_1["ci_97.5"] == diff_2["ci_97.5"]
            assert diff_1["mean_diff"] == diff_2["mean_diff"]

    def test_different_seed_changes_draw(
        self,
        gt_map: dict[str, sv.Detections],
        pred_maps: dict[str, dict[str, sv.Detections]],
    ) -> None:
        boot_seed0 = run_bootstrap(gt_map, pred_maps, n_boot=50, seed=0)
        boot_seed1 = run_bootstrap(gt_map, pred_maps, n_boot=50, seed=1)

        # model_b is sensitive to which images get drawn (empty on odd
        # images), so a different seed should change its bootstrap samples.
        assert not np.array_equal(
            boot_seed0["model_b::mAP_50_95"], boot_seed1["model_b::mAP_50_95"]
        )

    def test_paired_draw_shared_across_models(self, gt_map: dict[str, sv.Detections]) -> None:
        """Two identical models must get byte-identical per-iteration samples.

        This is only guaranteed if the SAME draw is used for every model
        within an iteration (paired bootstrap).
        """
        model = {filename: _det([[100, 100, 300, 400]], [0], [0.9]) for filename in gt_map}
        pred_maps = {"model_a": model, "model_b": dict(model)}

        boot = run_bootstrap(gt_map, pred_maps, n_boot=20, seed=7)

        assert np.array_equal(boot["model_a::mAP_50_95"], boot["model_b::mAP_50_95"])


class TestParallelBootstrapIsByteIdentical:
    """Parallelizing the per-iteration scoring must not move any digit.

    The draws are precomputed serially from one rng stream and executor.map
    preserves order, so run_bootstrap must return byte-identical arrays (and
    hence CIs) for max_workers=1 vs 4 vs 10. These cases pass max_workers
    EXPLICITLY (>1) so they actually spawn processes -- exercising the macOS
    'spawn' pickling path (module-level worker fn + initializer) -- rather than
    falling back to the small-n_boot serial auto-mode.
    """

    def test_arrays_identical_across_worker_counts(
        self,
        gt_map: dict[str, sv.Detections],
        pred_maps: dict[str, dict[str, sv.Detections]],
    ) -> None:
        serial = run_bootstrap(gt_map, pred_maps, n_boot=40, seed=0, max_workers=1)
        par4 = run_bootstrap(gt_map, pred_maps, n_boot=40, seed=0, max_workers=4)
        par10 = run_bootstrap(gt_map, pred_maps, n_boot=40, seed=0, max_workers=10)

        assert set(serial) == set(par4) == set(par10)
        for key in serial:
            # Byte-identical: exact array equality, not approximate.
            assert np.array_equal(serial[key], par4[key])
            assert np.array_equal(serial[key], par10[key])

    def test_report_cis_identical_across_worker_counts(
        self,
        gt_map: dict[str, sv.Detections],
        pred_maps: dict[str, dict[str, sv.Detections]],
    ) -> None:
        serial = run_bootstrap(gt_map, pred_maps, n_boot=40, seed=0, max_workers=1)
        par = run_bootstrap(gt_map, pred_maps, n_boot=40, seed=0, max_workers=4)
        report_serial = build_report(gt_map, pred_maps, serial, n_boot=40, seed=0)
        report_par = build_report(gt_map, pred_maps, par, n_boot=40, seed=0)

        for model in pred_maps:
            for metric in ("mAP_50_95", "mAP_50"):
                s = report_serial["per_model"][model][metric]
                p = report_par["per_model"][model][metric]
                assert s["ci_2.5"] == p["ci_2.5"]
                assert s["ci_97.5"] == p["ci_97.5"]
                assert s["bootstrap_mean"] == p["bootstrap_mean"]
                assert s["bootstrap_std"] == p["bootstrap_std"]

        pair_key = "model_a minus model_b"
        for metric in ("mAP_50_95", "mAP_50"):
            s = report_serial["pairwise"][pair_key][metric]
            p = report_par["pairwise"][pair_key][metric]
            assert s["ci_2.5"] == p["ci_2.5"]
            assert s["ci_97.5"] == p["ci_97.5"]
            assert s["point_diff"] == p["point_diff"]
            assert s["ci_excludes_zero"] == p["ci_excludes_zero"]


class TestBuildReport:
    """Structural tests for build_report's output shape."""

    def test_report_has_expected_keys(
        self,
        gt_map: dict[str, sv.Detections],
        pred_maps: dict[str, dict[str, sv.Detections]],
    ) -> None:
        boot = run_bootstrap(gt_map, pred_maps, n_boot=20, seed=0)
        report = build_report(gt_map, pred_maps, boot, n_boot=20, seed=0)

        assert report["config"]["n_boot"] == 20
        assert report["config"]["seed"] == 0
        assert report["config"]["n_images"] == len(gt_map)
        assert set(report["config"]["models"]) == {"model_a", "model_b"}

        for model in ("model_a", "model_b"):
            for metric in ("mAP_50_95", "mAP_50"):
                stats = report["per_model"][model][metric]
                assert "point_estimate" in stats
                assert "bootstrap_mean" in stats
                assert "bootstrap_std" in stats
                assert "ci_2.5" in stats
                assert "ci_97.5" in stats

        pair = report["pairwise"]["model_a minus model_b"]["mAP_50_95"]
        assert "point_diff" in pair
        assert "mean_diff" in pair
        assert "ci_excludes_zero" in pair
        # model_a strictly outperforms model_b (perfect on all vs half-empty).
        assert pair["point_diff"] > 0
