"""Assemble the MkDocs source tree (``site_src/``) for the GitHub Pages site.

The published prose lives in three places in the repo — ``README.md`` (root),
the generated reports under ``benchmarks/basketball/reports/``, and the docs
under ``docs/`` — so MkDocs (whose ``docs_dir`` must be a single tree) cannot
point at them directly. This script copies them into a flat ``site_src/`` tree
and rewrites the repo-relative cross-document links to their site locations, so
the same Markdown that renders on GitHub also renders correctly on the site.

``site_src/`` is generated (gitignored); it is rebuilt from scratch on every
run. The report Markdown itself is never edited — the reports remain
generator-emitted from the committed results files (REPORT-01); this only
stages copies for rendering.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_SRC = REPO_ROOT / "site_src"
REPORTS = REPO_ROOT / "benchmarks" / "basketball" / "reports"
DOCS = REPO_ROOT / "docs"

# (source file, destination relative to site_src/, [(old_link, new_link), ...])
_Copy = tuple[Path, str, list[tuple[str, str]]]

_PLAN: list[_Copy] = [
    (
        REPO_ROOT / "README.md",
        "index.md",
        [
            (
                "benchmarks/basketball/reports/FINAL_COMPARISON_640.md",
                "reports/FINAL_COMPARISON_640.md",
            ),
            (
                "benchmarks/basketball/reports/VLM_VS_FINETUNED.md",
                "reports/VLM_VS_FINETUNED.md",
            ),
            ("docs/methodology.md", "methodology.md"),
        ],
    ),
    (
        REPORTS / "FINAL_COMPARISON_640.md",
        "reports/FINAL_COMPARISON_640.md",
        [("../../../docs/methodology.md", "../methodology.md")],
    ),
    (
        REPORTS / "VLM_VS_FINETUNED.md",
        "reports/VLM_VS_FINETUNED.md",
        [("../../../docs/methodology.md", "../methodology.md")],
    ),
    (
        DOCS / "methodology.md",
        "methodology.md",
        [
            (
                "../benchmarks/basketball/reports/FINAL_COMPARISON_640.md",
                "reports/FINAL_COMPARISON_640.md",
            ),
            (
                "../benchmarks/basketball/reports/VLM_VS_FINETUNED.md",
                "reports/VLM_VS_FINETUNED.md",
            ),
        ],
    ),
    (DOCS / "provenance" / "training-runs.md", "provenance/training-runs.md", []),
    (DOCS / "provenance" / "artifact-tracker.md", "provenance/artifact-tracker.md", []),
    (DOCS / "provenance" / "gcs-manifest.md", "provenance/gcs-manifest.md", []),
]


def stage() -> None:
    """Rebuild ``site_src/`` from the committed prose, rewriting cross-links."""
    if SITE_SRC.exists():
        shutil.rmtree(SITE_SRC)
    SITE_SRC.mkdir(parents=True)

    for source, dest_rel, link_rewrites in _PLAN:
        if not source.is_file():
            msg = f"stage_docs: expected source file is missing: {source}"
            raise FileNotFoundError(msg)
        text = source.read_text(encoding="utf-8")
        for old, new in link_rewrites:
            if old not in text:
                logger.warning(
                    f"stage_docs: link '{old}' not found in {source.name} "
                    "(report content may have changed) -- skipping rewrite"
                )
            text = text.replace(old, new)
        dest = SITE_SRC / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        logger.info(f"staged {source.relative_to(REPO_ROOT)} -> site_src/{dest_rel}")

    logger.info(f"staged {len(_PLAN)} pages into {SITE_SRC}")


if __name__ == "__main__":
    stage()
