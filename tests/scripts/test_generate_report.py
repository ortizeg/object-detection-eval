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
    # Real results dir, but fresh (empty) report AND docs dirs: every registered
    # report has no .md yet, so main skips them all and exits 0. Both dirs are
    # redirected so the assertion stays about skipping, not about whether the
    # committed documents happen to be in sync.
    exit_code = generate_report.main(
        [
            "--check",
            "--report-dir",
            str(tmp_path),
            "--docs-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0


def test_main_unknown_report_id_errors(tmp_path: Path) -> None:
    exit_code = generate_report.main(
        [
            "--check",
            "--report",
            "does_not_exist",
            "--report-dir",
            str(tmp_path),
            "--docs-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 2


def test_dataset_page_is_registered_and_reads_only_committed_json(tmp_path: Path) -> None:
    """The dataset page must be in the registry, and must not need the dataset.

    The raw dataset is absent on the CI machine, so a dataset slot that reached
    for it would make the whole drift gate unrunnable there. Rendering it here —
    in a suite that has no dataset — is what proves it does not.
    """
    specs = generate_report.build_registry(
        _REPO_ROOT / "benchmarks" / "basketball" / "results",
        tmp_path,
    )
    dataset = next(s for s in specs if s.report_id == "dataset")
    assert dataset.md_path == _REPO_ROOT / "docs" / "dataset.md"
    for slot in dataset.slots:
        assert slot.render()  # renders from committed JSON alone


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


def test_cpu_latency_section_skips_gracefully_when_results_absent(
    tmp_path: Path,
) -> None:
    # Robustness invariant (repo-state-independent): with the measured CPU
    # results files absent, the CPU/edge section renders the fixed notice rather
    # than a table and never raises, so --check stays green before those files
    # are committed. The present-case (real CPU table) is covered by
    # test_committed_reports_are_not_drifted. Tested hermetically against a tmp
    # dir so it holds whether or not the CPU results are committed.
    rendered = generate_report._render_cpu_latency(
        tmp_path / "cpu_e2e_conf025.json",
        tmp_path / "cpu_e2e_conf001.json",
    )
    assert rendered == generate_report._CPU_LATENCY_ABSENT_NOTICE


def test_committed_reports_are_not_drifted() -> None:
    """The live REPORT-01 CI gate, covering every registered document.

    Registry-driven rather than an explicit list, so a newly registered
    document (e.g. the dataset page) is covered the moment it is added.
    """
    report_dir = _REPO_ROOT / "benchmarks" / "basketball" / "reports"
    specs = generate_report.build_registry(
        _REPO_ROOT / "benchmarks" / "basketball" / "results",
        report_dir,
    )
    present = [s for s in specs if s.md_path.is_file()]
    if not present:
        pytest.skip("no committed reports yet; drift guard is dormant until Wave 2")
    assert generate_report._run(present, check=True, write=False) == 0
