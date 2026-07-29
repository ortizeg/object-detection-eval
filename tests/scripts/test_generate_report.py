"""Offline drift tests for scripts/generate_report.py (REPORT-01).

Loaded by file path (``scripts/`` is not a package), mirroring
``test_run_benchmark``. Exercises the render/inject/compare code paths against
the committed test fixtures (not the real 94-image data or the bootstrap run),
so the suite stays fast and torch-free in the default CI selection:

- ``--check`` on a doc that matches its results file exits 0; a hand-edited cell
  exits nonzero (the anti-drift gate).
- ``--write`` injects the table; a second ``--write`` is a byte-for-byte no-op.
- ``main`` skips a registered report whose ``.md`` does not yet exist (safe to
  run in Wave 1 before the reports are authored).

A dormant guard runs ``--check`` against the real committed reports IF they
exist, becoming the live REPORT-01 CI enforcement once the Wave-2 reports land.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from object_detection_eval.report import load_accuracy_results, primary_7model_table

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "generate_report.py"
_ACCURACY_FIXTURE = _REPO_ROOT / "tests" / "report" / "fixtures" / "accuracy_merged5.json"

_DOC_TEMPLATE = (
    "# Report\n\nIntro prose.\n\n"
    "<!-- TABLE:primary_7model START -->\n"
    "stale placeholder\n"
    "<!-- TABLE:primary_7model END -->\n\n"
    "Outro prose.\n"
)


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("generate_report", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate_report = _load_module()


def _fixture_spec(md_path: Path) -> object:
    """A single-slot report spec backed by the committed accuracy fixture."""
    slot = generate_report.Slot(
        "primary_7model",
        lambda: primary_7model_table(load_accuracy_results(_ACCURACY_FIXTURE)),
    )
    return generate_report.ReportSpec(
        report_id="fixture_report",
        md_path=md_path,
        slots=[slot],
    )


def test_write_then_check_is_in_sync(tmp_path: Path) -> None:
    doc = tmp_path / "report.md"
    doc.write_text(_DOC_TEMPLATE, encoding="utf-8")
    spec = _fixture_spec(doc)

    assert generate_report._run([spec], check=False, write=True) == 0
    # The rendered table replaced the placeholder.
    assert "stale placeholder" not in doc.read_text()
    assert "YOLO26m" in doc.read_text()
    # --check now finds no drift.
    assert generate_report._run([spec], check=True, write=False) == 0


def test_write_is_idempotent(tmp_path: Path) -> None:
    doc = tmp_path / "report.md"
    doc.write_text(_DOC_TEMPLATE, encoding="utf-8")
    spec = _fixture_spec(doc)

    generate_report._run([spec], check=False, write=True)
    first = doc.read_text()
    generate_report._run([spec], check=False, write=True)
    assert doc.read_text() == first


def test_check_fails_on_hand_edited_cell(tmp_path: Path) -> None:
    doc = tmp_path / "report.md"
    doc.write_text(_DOC_TEMPLATE, encoding="utf-8")
    spec = _fixture_spec(doc)

    generate_report._run([spec], check=False, write=True)
    tampered = doc.read_text().replace("0.716", "0.999")
    assert tampered != doc.read_text()  # sanity: the cell existed
    doc.write_text(tampered, encoding="utf-8")

    assert generate_report._run([spec], check=True, write=False) == 1


def test_main_skips_report_without_document(tmp_path: Path) -> None:
    # Real results dir, but a fresh (empty) report dir: every registered report
    # has no .md yet, so main skips them all and exits 0.
    exit_code = generate_report.main(
        [
            "--check",
            "--report-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0


def test_main_unknown_report_id_errors(tmp_path: Path) -> None:
    exit_code = generate_report.main(
        ["--check", "--report", "does_not_exist", "--report-dir", str(tmp_path)]
    )
    assert exit_code == 2


_CPU_FIXTURE_DIR = _REPO_ROOT / "tests" / "report" / "fixtures"


def test_render_cpu_latency_absent_returns_notice(tmp_path: Path) -> None:
    # Both files missing -> the section renders the deterministic notice, so the
    # drift gate passes GT-free before the orchestrator commits the real CPU run.
    out = generate_report._render_cpu_latency(
        tmp_path / "cpu_e2e_conf025.json", tmp_path / "cpu_e2e_conf001.json"
    )
    assert out == generate_report._CPU_LATENCY_ABSENT_NOTICE
    assert "not committed yet" in out


def test_render_cpu_latency_present_returns_table() -> None:
    out = generate_report._render_cpu_latency(
        _CPU_FIXTURE_DIR / "cpu_e2e_conf025.json",
        _CPU_FIXTURE_DIR / "cpu_e2e_conf001.json",
    )
    assert "Δ (NMS blow-up)" in out
    assert "dense + Python NMS" in out
    assert "not committed yet" not in out


def test_check_passes_with_cpu_results_absent() -> None:
    # The critical CI invariant: --check on the committed reports exits 0 with the
    # real CPU results files ABSENT (they are not committed until the orchestrator
    # runs the CPU benchmark). Simulate CI by pointing at the real results dir.
    results_dir = _REPO_ROOT / "benchmarks" / "basketball" / "results"
    assert not (results_dir / "latency" / "cpu_e2e_conf025.json").exists()
    report_dir = _REPO_ROOT / "benchmarks" / "basketball" / "reports"
    specs = generate_report.build_registry(results_dir, report_dir)
    present = [s for s in specs if s.md_path.is_file()]
    if not present:
        pytest.skip("no committed reports yet")
    assert generate_report._run(present, check=True, write=False) == 0


def test_committed_reports_are_not_drifted() -> None:
    """Dormant REPORT-01 CI gate: enforced once the Wave-2 reports exist."""
    report_dir = _REPO_ROOT / "benchmarks" / "basketball" / "reports"
    specs = generate_report.build_registry(
        _REPO_ROOT / "benchmarks" / "basketball" / "results",
        report_dir,
    )
    present = [s for s in specs if s.md_path.is_file()]
    if not present:
        pytest.skip("no committed reports yet; drift guard is dormant until Wave 2")
    assert generate_report._run(present, check=True, write=False) == 0
