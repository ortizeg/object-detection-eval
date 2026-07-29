# Phase 7: Reports & Docs - Research

**Researched:** 2026-07-29
**Domain:** Markdown report generation from committed results data (internal architecture, no new external tech)
**Confidence:** HIGH

## Summary

Phase 7 has no new external library or framework decision to make — the entire dependency
surface (pydantic, loguru, PyYAML, `supervision`, pytest) was already adopted in Phase 2 and
is already a core (non-`[vlm]`, non-`[trt]`) dependency. The work is 100% internal: build a
committed, torch-free **report generator** that reads already-committed (or newly-produced)
results JSON and emits markdown tables, wire it so CI can catch drift, and then write the prose
around those generated tables.

The single biggest finding of this research is **the accuracy-results gap named in the task
brief is real and has a clean, low-risk fix**: `compute_metrics()` (Phase 2) already returns
`per_class_ap50` alongside the three mAP variants, and `build_report()` (Phase 2's bootstrap
module) already returns the exact `per_model`/`pairwise` JSON shape that
`.deploy_comparison/bootstrap_5c_test_7models.json` uses as its anchor. Neither
`scripts/run_benchmark.py` nor `scripts/run_bootstrap_gate.py` currently writes its result to
disk — both only print a table and exit 0/1. Phase 7 needs a `--write-results` (or equivalent)
flag added to each, run once by the executor (mirroring exactly how `results/vlm/*.json` and
`results/latency/*.json` were produced in Phases 5 and 6: executor-run, not CI-wired, committed
as provenance), producing `benchmarks/basketball/results/accuracy/*.json`. **All artifacts this
requires (source_repo `.deploy_comparison/`, the external `yolox` root, and the basketball test
GT) are present on this machine right now** — end2end mode can run today with no new
dependency, no rented GPU, and no missing file.

The VLM domain has no equivalent gap: all 6 `results/vlm/*.json` files are already committed
(prediction dumps, `git ls-files` confirms `grounding_dino.json`/`omdet_turbo.json`/
`owlv2.json` are tracked despite an earlier plan summary noting an intent to gitignore them —
that intent was evidently reversed before commit). The generator can call
`compute_metrics(gt_map, pred_map, id_to_name)` directly on these files to recompute per-class
AP50 for the rim-collapse / zero-AP-ball-referee finding, rather than transcribing numbers from
the (now-superseded) source-repo `EVAL_REPORT.md`.

**Primary recommendation:** Add a `report/` package under `src/object_detection_eval/` (an
already-anticipated architectural slot per `ROADMAP.md`'s own Phase 2 overview: "the 1671-line
task splits into `data/`, `metrics/`, `inference/`, `report/`" — only `report/` was deferred to
this phase) with typed loaders for each results-file shape and pure markdown-table renderers,
plus a thin `scripts/generate_report.py` CLI with `--check` (CI-wired drift gate) and `--write`
modes that inject generated tables between HTML-comment markers inside otherwise hand-authored
`FINAL_COMPARISON_640.md` / `VLM_VS_FINETUNED.md`. Two new executor-run scripts extensions
(`--write-results` on `run_benchmark.py` and `run_bootstrap_gate.py`) close the accuracy-data
gap first; the generator is built and tested against those files second; the two reports and
`docs/methodology.md`/`README.md` updates come last.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Accuracy scoring (mAP, per-class AP, CIs) | Data/Metrics tier (`src/object_detection_eval/metrics/`) | Scripts tier (`scripts/run_benchmark.py`, `scripts/run_bootstrap_gate.py`) | Metrics computation already lives in the public, tested library API; scripts only orchestrate manifest-driven runs and (new) persistence. No new computation belongs in the report tier. |
| Results persistence (JSON on disk) | Scripts tier | — | `run_benchmark.py`/`run_bootstrap_gate.py` are the only code that has both the manifest and the live metrics dict in hand; writing is a thin addition to their existing `main()`, not a new component. |
| Markdown table rendering | New `report/` library tier (`src/object_detection_eval/report/`) | Scripts tier (`scripts/generate_report.py` CLI) | Rendering is pure, testable, torch-free logic — belongs in `src/` per this repo's convention (business logic in `src/`, thin CLIs in `scripts/`), not embedded in a script. |
| Report document assembly (prose + injected tables) | Committed markdown files (`benchmarks/basketball/reports/*.md`) | `report/` tier (marker injection) | Prose (the preprocessing-finding lede, fairness-audit narrative, per-class failure analysis) is authored content; only the numeric tables are machine-owned. The marker-injection pattern is what keeps these two concerns cleanly separated in one file. |
| CI drift detection | CI tier (`.github/workflows/test.yml` + `pixi run test`) | Scripts tier (`--check` mode) | REPORT-01's guarantee ("no published number can drift from the data that produced it") is only real if something enforces it automatically — a `--check` mode callable from the existing default (torch-free) CI job is the mechanism. |
| Model metadata (license, params, backbone, style) | Model Registry tier (`registry/*.yaml` via `ModelCard`) | Report tier (read-only join) | This data is already schema-validated (REG-01..06) and human-curated at a slower cadence than benchmark numbers; joining it into report tables (not re-deriving it) avoids duplicating REG's source of truth. |

## Standard Stack

### Core

No new dependency is needed. Everything the generator requires is already a core
(non-optional) dependency, verified against the installed/locked versions:

| Library | Version (locked) | Purpose | Why Standard (here) |
|---------|---------|---------|--------------|
| `pydantic` | `>=2.0` (already core) | Typed models for each results-file shape (`AccuracyResult`, `BootstrapReport`, `LatencyResult`, `VLMRow`) | Every other manifest/results-adjacent structure in this repo (`ManifestEntry`, `ModelCard`) is a frozen pydantic model; consistency with `run_benchmark.py`/`run_bootstrap_gate.py`/`registry/model_card.py`. |
| `loguru` | already core | All CLI output in the generator (`--check` failures, `--write` progress) | CORE-09 mandates loguru-only output, ruff `T20` (no `print`) is already enforced repo-wide; the generator is new code and must comply from day one. |
| `PyYAML` | already core | Reading `registry/*.yaml` model cards for license/params/backbone/style joins | Already how `ModelCard`/`Manifest` loaders read YAML (`yaml.safe_load`). |
| `supervision` | `==0.29.1` (pinned, already core) | Re-deriving VLM per-class AP directly from committed prediction JSON via `compute_metrics` | Reuses the exact scoring path already pinned and validated (Phase 2, CORE-02/04) — the report tier must not re-implement mAP. |
| `pytest` | already dev dep | Golden-value test: a specific rendered table cell equals the results-file value | Matches the existing test convention (`tests/scripts/test_run_benchmark.py` etc.) — offline, no external artifacts. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| stdlib `re` | stdlib | Locating `<!-- TABLE:name START -->`/`END` marker pairs for in-place injection | No markdown templating library is warranted for ~5 table slots across 2 files; a marker-regex replace is simpler, has zero new dependency surface, and is trivially unit-testable. |
| stdlib `argparse` | stdlib | `scripts/generate_report.py` CLI (`--check`, `--write`, `--report {final_comparison,vlm_vs_finetuned,both}`) | Matches every other script's CLI shape (`run_benchmark.py`, `run_vlm_benchmark.py`, `run_latency.py`). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Marker-comment injection into hand-authored `.md` | Jinja2 full-document templates | Jinja2 would require the ENTIRE report body (including all hand-written prose — the preprocessing lede, fairness audit narrative) to live inside a template, which fights REPORT-02/03's requirement for substantial curated prose and risks the generator "owning" content it shouldn't. Marker injection keeps prose under normal git diff review while still making tables un-editable-by-hand. Not adopted. |
| pydantic models for results JSON | Raw `dict[str, Any]` + manual key access | The repo's own convention (every manifest, every model card) is a frozen pydantic model with validation; raw dicts would be inconsistent and lose the "fails loudly with a named error" property REG-02 established elsewhere. Not adopted. |
| A new `results/accuracy/*.json` write path in a brand-new script | Extending `run_benchmark.py`/`run_bootstrap_gate.py` in place | A new script would duplicate the entire manifest-loading, precondition-checking, and metrics-computation logic already in these two files for zero benefit — the write is a 5-10 line addition to each script's existing `main()`. Not adopted (see Architecture Patterns). |

**Installation:** None required — no `pyproject.toml`/`pixi.toml` change for REPORT-01..05 itself.

**Version verification:** N/A — no new package. `supervision==0.29.1` and `pydantic>=2.0` are
already pinned/installed in this environment (confirmed by reading `pyproject.toml` directly,
Phase 2/4 already validated their behavior against this exact `supervision` pin in
`docs/methodology.md`).

## Package Legitimacy Audit

**Not applicable — this phase installs no new external packages.** Every library the generator
needs (`pydantic`, `loguru`, `pyyaml`, `supervision`, stdlib `re`/`argparse`/`json`) is already a
verified, in-use core dependency from prior phases. No `npm view`/`pip index versions` check is
needed because nothing new is being added to `pyproject.toml` or `pixi.toml`.

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │  EXECUTOR-RUN, NOT CI-WIRED (mirrors Phase 5/6)      │
                    │                                                       │
  registry ONNX ──▶ │  run_benchmark.py --mode end2end                    │
  (or source_repo   │    --write-results results/accuracy/                │
  .deploy_comparison│    reproduction_640_{merged5,raw10}.json             │
  for dev/CI-mirror)│  run_bootstrap_gate.py                               │
                    │    --write-results results/accuracy/                │
                    │    bootstrap_7models.json                            │
                    └──────────────────────┬────────────────────────────────┘
                                            │ commit as provenance
                                            ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  COMMITTED RESULTS (git-tracked, read-only inputs to the generator)      │
  │                                                                           │
  │  results/accuracy/reproduction_640_merged5.json  (mAP + per-class, 5c)  │
  │  results/accuracy/reproduction_640_raw10.json    (mAP + per-class, 10c) │
  │  results/accuracy/bootstrap_7models.json         (CIs + pairwise sig)   │
  │  results/vlm/{gemini,owlv2,grounding_dino,omdet_turbo,                  │
  │               florence2,smolvlm2}.json           (predictions, Phase 5) │
  │  results/latency/{uniform_e2e,trt_fp16_gpuonly,                         │
  │                    trt_fp16_toboxes}.json         (Phase 6, w/ honest-  │
  │                                                    label reproducibility│
  │                                                    field)               │
  │  registry/*.yaml                                 (license/params/      │
  │                                                    backbone/style, REG) │
  └───────────────────────────┬───────────────────────────────────────────┘
                               │ read-only, torch-free
                               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CI-WIRED, TORCH-FREE (default `pixi run test` suite)                    │
  │                                                                           │
  │  src/object_detection_eval/report/                                       │
  │    loaders.py   -- typed pydantic readers per results-file shape         │
  │    tables.py    -- pure functions: dict -> markdown table string         │
  │    inject.py    -- marker-comment find/replace into a target .md file    │
  │                                                                           │
  │  scripts/generate_report.py --check   (CI: fails if committed .md drifts │
  │                                         from committed results/)         │
  │  scripts/generate_report.py --write   (executor: regenerate tables)      │
  └───────────────────────────┬───────────────────────────────────────────┘
                               │ inject between <!-- TABLE:x START/END -->
                               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  HAND-AUTHORED PROSE + GENERATED TABLES (git-tracked .md, human-reviewed)│
  │                                                                           │
  │  benchmarks/basketball/reports/FINAL_COMPARISON_640.md   (REPORT-02)    │
  │  benchmarks/basketball/reports/VLM_VS_FINETUNED.md        (REPORT-03)    │
  │  docs/methodology.md                                       (REPORT-04)  │
  │  README.md                                                  (REPORT-05) │
  └─────────────────────────────────────────────────────────────────────────┘
```

A reader tracing the primary use case: an executor runs the two `--write-results` extensions
once (top box) → the resulting JSON is committed alongside the already-committed VLM/latency/
registry data (second box) → `scripts/generate_report.py --write` reads all of it and injects
markdown tables into the two reports plus (optionally) methodology/README fragments (third box)
→ the final `.md` files ship with generated tables and hand-written narrative side by side
(bottom box). `--check` runs the same read+render step in CI and diffs against the committed
`.md`, catching any hand-edited table or any results file that changed without regenerating.

### Recommended Project Structure

```
src/object_detection_eval/
└── report/
    ├── __init__.py       # public API: render_accuracy_table, render_per_class_table,
    │                      # render_ci_table, render_latency_section, render_vlm_table,
    │                      # inject_tables — torch-free, no imports beyond pydantic/loguru/
    │                      # supervision/stdlib (VLM rendering reads PRE-COMPUTED prediction
    │                      # JSON through supervision/compute_metrics, never imports inference/vlm/*)
    ├── loaders.py         # AccuracyResult, BootstrapReport, LatencyResult pydantic models +
    │                      # load_accuracy_results(), load_bootstrap_report(),
    │                      # load_latency_results(), load_vlm_metrics() (the last one calls
    │                      # metrics.bootstrap.load_predictions + metrics.detection_map.
    │                      # compute_metrics directly on results/vlm/*.json)
    ├── tables.py          # pure fn(loaded-model) -> markdown string, one fn per table:
    │                      # primary_7model_table, ci_table, per_class_table(taxonomy),
    │                      # fairness_prose is NOT here (hand-authored), latency_table,
    │                      # vlm_summary_table, vlm_per_class_table
    └── inject.py          # find "<!-- TABLE:{name} START -->"..."<!-- TABLE:{name} END -->"
                            # in a target file, replace the interior, preserve everything else

scripts/
└── generate_report.py     # CLI: --report {final_comparison,vlm_vs_finetuned,methodology,all}
                            # --check (exit 1 on drift, prints unified diff via loguru)
                            # --write (regenerate in place)

benchmarks/basketball/
├── results/
│   └── accuracy/           # NEW in this phase
│       ├── reproduction_640_merged5.json
│       ├── reproduction_640_raw10.json
│       └── bootstrap_7models.json
└── reports/                # NEW in this phase
    ├── FINAL_COMPARISON_640.md
    └── VLM_VS_FINETUNED.md

tests/
└── report/
    ├── test_loaders.py     # golden-value: a loaded field equals a fixture JSON's raw value
    ├── test_tables.py      # a specific rendered cell (e.g. "0.716") appears in the YOLO26m
    │                        # row of the rendered primary table, sourced from a fixture, NOT
    │                        # the real committed results file (keeps the test hermetic)
    └── test_inject.py      # marker replace is idempotent and leaves surrounding prose intact
```

### Pattern 1: Executor-run "write, don't compute-in-CI" for expensive/external-artifact results

**What:** Any script that needs external, gitignored, or slow-to-produce artifacts (ONNX
weights, a rented GPU, a ~4.5h serial-CPU bootstrap) writes its result to a committed JSON file
once, run by a human/executor outside CI; downstream code (the report generator, tests) only
ever reads that committed JSON.

**When to use:** Already the established pattern for `results/vlm/*.json` (Phase 5,
`run_vlm_benchmark.py`, executor-run on a rented RTX 4090) and `results/latency/*.json`
(Phase 6, `run_latency.py`/`build_trt_engines.py`, executor-run on a rented T4). REPORT-01's
accuracy gap is the same shape of problem and should use the same pattern, not a new one.

**Example (the extension to add, following `run_vlm_benchmark.py`'s existing write path):**
```python
# Source: scripts/run_vlm_benchmark.py's existing results-file write (Phase 5, already committed
# code) — the SAME pattern applies to run_benchmark.py, which currently only prints and never
# writes.
def _write_results_file(path: Path, per_model: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(per_model, f, indent=2)
    logger.info(f"Wrote accuracy results to {path}")
```

### Pattern 2: Marker-comment injection, not full-file templating

**What:** A generated table lives between two HTML comments the generator recognizes by exact
string match; everything outside the markers is untouched, byte-for-byte, on every `--write`.

**When to use:** Any markdown document that mixes long-form hand-written narrative with a small
number of strictly-generated tables — exactly REPORT-02/03/04's shape (the preprocessing-finding
lede, the fairness-audit prose, the per-class failure-analysis narrative are all hand-authored;
only the numeric tables are generated).

**Example:**
```markdown
<!-- TABLE:primary_7model START -->
| Model | ... |
|---|---|
...
<!-- TABLE:primary_7model END -->
```
```python
# Source: new code, `report/inject.py` — no external reference, this is the pattern to build.
import re

_MARKER = re.compile(
    r"<!-- TABLE:{name} START -->\n.*?<!-- TABLE:{name} END -->",
    re.DOTALL,
)

def inject_table(doc: str, name: str, table_markdown: str) -> str:
    pattern = re.compile(
        rf"<!-- TABLE:{re.escape(name)} START -->\n.*?<!-- TABLE:{re.escape(name)} END -->",
        re.DOTALL,
    )
    replacement = f"<!-- TABLE:{name} START -->\n{table_markdown}\n<!-- TABLE:{name} END -->"
    new_doc, count = pattern.subn(replacement, doc)
    if count != 1:
        msg = f"expected exactly one marker pair for {name!r}, found {count}"
        raise ValueError(msg)
    return new_doc
```

### Pattern 3: Registry cards as the join source for non-benchmark metadata

**What:** License, param count, backbone, and architecture style come from `registry/*.yaml`
(already schema-validated by `ModelCard`, REG-01..06), joined into report tables by model name —
never re-typed by hand into the report generator.

**When to use:** Any report column that is architecture/licensing metadata rather than a
benchmark measurement. `registry/yolo26m_640.yaml`'s `evaluations` block already carries
`map5095_5c: 0.716` etc. as a **secondary cross-check**, not the primary source — the primary
source for benchmark numbers is `results/accuracy/*.json` (REPORT-01's actual requirement); a
generator-side assertion that the two agree (registry card vs. freshly-computed results file) is
a cheap extra integrity check worth adding as a test, not a requirement for the table itself.

### Anti-Patterns to Avoid

- **Hand-transcribing per-class AP for the VLM report from `EVAL_REPORT.md`:** That file is the
  *training* repo's legacy/superseded report (superseded even within that repo by
  `EVAL_REPORT_FINAL.md` for the medium-model claim) and its VLM per-class numbers predate the
  current harness's exact filter pipeline (`remap → area_outliers → single_best_per_class`,
  Phase 5). The committed `results/vlm/*.json` prediction dumps let the generator recompute
  per-class AP directly and correctly from this repo's own scoring path — use them as the
  numbers, use `EVAL_REPORT.md`'s prose only as a sanity-check reference for the *shape* of the
  finding (rim collapse, zero-AP ball/referee for two of the five methods).
- **A new script duplicating `run_benchmark.py`'s manifest/precondition logic just to "write
  results":** the write is a few lines inside the existing `main()`; a parallel script would
  double-maintain the detector-factory dict, the precondition assertions, and the taxonomy
  resolution for no benefit.
- **Full Jinja2 templating of the entire report body:** see Alternatives Considered — it would
  make prose editing indistinguishable from a template change in diffs, and there's no need
  (only ~5-7 table slots total across both reports).
- **Silently widening the `[trt]`/`[vlm]` boundary:** the generator must stay importable and
  runnable with `pixi run test` (the CORE-08 torch-free default env) since it only reads JSON —
  do not accidentally import `inference/vlm/*` or `inference/onnx.py`'s ONNX-runtime path from
  `report/`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-class AP for the VLM report | A bespoke per-class scorer over `results/vlm/*.json` | `metrics.detection_map.compute_metrics(gt_map, pred_map, id_to_name)['per_class_ap50']`, feeding `pred_map = metrics.bootstrap.load_predictions(path)` | Already public, typed, tested (Phase 2 CORE-02); it is the exact function the reproduction/VLM gates already trust. |
| Paired-bootstrap CIs for the accuracy report | A one-off CI script for the report | `metrics.bootstrap.run_bootstrap` + `build_report` (already produces the anchor-matching `per_model`/`pairwise` JSON shape) | Reproduces `.deploy_comparison/bootstrap_5c_test_7models.json`'s exact structure; Phase 4 (REPRO-03) already validated it byte-for-byte against the anchor. |
| Markdown table formatting | A markdown-table library (e.g. `tabulate`) | Plain f-string joins over a list of rows | ~5-7 fixed-shape tables total; a dependency for this is disproportionate, and every row already comes from a typed pydantic model so alignment/formatting is trivial to hand-write and unit-test. |
| Model metadata for report columns (license, params, backbone) | Re-typing values into the report generator or a new metadata file | `registry.model_card.load_model_card()` / the already-validated `registry/*.yaml` cards | REG-01..06 already made this the single source of truth with load-time validation; duplicating it anywhere else reintroduces the exact "published number can drift from the data" risk REPORT-01 exists to close. |

**Key insight:** every piece of computation this phase needs (mAP, per-class AP, bootstrap CIs,
model metadata validation) was already built, tested, and gated in Phases 2-6. Phase 7's own
code surface should be almost entirely *rendering* — read typed data, emit markdown — not new
metric logic. Any table cell that required new math to produce would be a red flag that the
generator has drifted into re-implementing something Phase 2/4 already owns.

## Common Pitfalls

### Pitfall 1: Treating the latency §6 second-run numbers as if they were reproduced

**What goes wrong:** A report generator (or a human writing the latency section) reads
`results/latency/trt_fp16_toboxes.json`'s `models[]` array and renders those numbers as "the"
latency table, silently dropping the `reproducibility` object at the top of the file.

**Why it happens:** The `models[]` array *does* contain a full second measurement (2026-07-29,
a different vast.ai T4) and looks like a normal results table — nothing in its shape forces a
reader to notice it is explicitly labeled non-portable.

**How to avoid:** The generator's latency renderer MUST read `reproducibility.label` (exact
string: `"manually measured 2026-07-21, not reproducible from this repo"`) and
`reproducibility.status` (`"manually_measured"`) and prepend/caption the rendered table with
that label verbatim, plus the `note`/`second_run.outcome` field explaining the method-reproduces-
but-absolute-numbers-don't-port finding (LAT-04's honest-label resolution, `06-03-SUMMARY.md`).
The §6 table in `FINAL_COMPARISON_640.md` should present the **original published band (4.0–7.1
ms, `source_band_ms_2026_07_21`)** as the headline figure with the honest-label caveat attached,
NOT the second-run absolute numbers re-labeled as if they were the reproduction.

**Warning signs:** A rendered latency table with no caption/footnote distinguishing "manually
measured, not reproducible from this repo" from a fully-reproduced number; any prose claiming
the second T4 run "confirms" the 4.0-7.1ms band (it explicitly does not — it confirms the
*method*, not the *absolute latency*).

### Pitfall 2: Scoring VLM per-class AP without `id_to_name`

**What goes wrong:** Calling `compute_metrics(gt_map, pred_map)` (omitting `id_to_name`) returns
`per_class_ap50` keyed by raw string class-id ("0", "1", ...) instead of class names
("player", "ball", ...), producing an unreadable or silently-mislabeled per-class table.

**Why it happens:** `id_to_name` is an optional third argument with a "defensive fallback" to
raw ids — the function does not error, so a missing argument fails silently, not loudly.

**How to avoid:** Always call `resolve_taxonomy("merged5")` to get `id_to_name` and pass it
explicitly: `compute_metrics(gt_map, pred_map, id_to_name)`. The `report/loaders.py` VLM loader
should hardcode this call, not leave it to each call site.

**Warning signs:** A per-class table with numeric-string column headers instead of class names.

### Pitfall 3: Sourcing the 10-class per-class table from the wrong taxonomy run

**What goes wrong:** `run_benchmark.py --mode end2end` (or `--mode from-predictions`) defaults to
`--taxonomy merged5`; running it once and trying to report both the 5-class AND 10-class
per-class tables (EVAL_REPORT_FINAL.md §3 and §4) from a single run silently produces wrong
10-class numbers (or a `KeyError`/mismatched class count), because merged5 and raw10 are
different `TaxonomySpec`s with different class counts and different `id_to_name` mappings.

**Why it happens:** The manifest (`reproduction_640.yaml`) and the CLI flag make it easy to run
once and forget the second taxonomy exists.

**How to avoid:** The `--write-results` extension to `run_benchmark.py` must be run twice — once
per taxonomy (`--taxonomy merged5 --write-results .../reproduction_640_merged5.json` and
`--taxonomy raw10 --write-results .../reproduction_640_raw10.json`) — producing two separate
committed files. `raw10`'s `player-layup-dunk` class has **zero test-set support** (0 instances,
per `EVAL_REPORT_FINAL.md` §4); the renderer must handle an undefined/absent AP for that class
(render as `—`, matching the source report) rather than crashing on a missing key or rendering
`0.000` (which would misleadingly imply the class was scored and failed, not simply absent).

**Warning signs:** A 10-class per-class table with `player-layup-dunk` rendered as `0.000` for
every model, or a generator crash on that row.

### Pitfall 4: Presenting "every adjacent pair is significant" as the headline claim

**What goes wrong:** The original `EVAL_REPORT_FINAL.md` prose over-claims "every adjacent pair
is significant" (§2). Phase 4 (REPRO-03) already traced this to a factual error: the anchor JSON
itself records **RTMDet-M vs DAMO-YOLO-M as a statistical tie** (`ci_excludes_zero: false`, CI
`[−0.0022, +0.0200]`) — 5 of 6 adjacent pairs are significant, not 6 of 6.

**Why it happens:** Copy-forwarding the source report's summary sentence without re-deriving it
from the actual per-pair CI data now sitting in `results/accuracy/bootstrap_7models.json`.

**How to avoid:** The generator must render pairwise significance as a **derived** column (
`ci_excludes_zero` from the bootstrap JSON, not a hand-typed "all significant" sentence), and the
report prose (hand-authored, around the generated table) must say "5 of 6 adjacent pairs
significant; RTMDet-M vs DAMO-YOLO-M is a statistical tie" — per `docs/methodology.md`'s already-
corrected framing and `04-03-SUMMARY.md`'s exact verified numbers below.

**Warning signs:** Report prose claiming "every adjacent pair is significant" without the
RTMDet-M/DAMO-YOLO-M tie caveat.

## Code Examples

Verified patterns from this repo's own already-committed source (all `[VERIFIED: local
codebase]` — read directly, not from external docs, since no external API is involved):

### Computing per-class AP for a VLM results file

```python
# Source: src/object_detection_eval/metrics/bootstrap.py::load_predictions +
# src/object_detection_eval/metrics/detection_map.py::compute_metrics (both already public/typed)
from pathlib import Path
from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.bootstrap import load_predictions
from object_detection_eval.metrics.detection_map import compute_metrics

name_to_id, id_to_name = resolve_taxonomy("merged5")
gt_map = load_coco_gt(Path("test/_annotations.coco.json"), name_to_id)
pred_map = load_predictions(Path("benchmarks/basketball/results/vlm/gemini.json"))
metrics = compute_metrics(gt_map, pred_map, id_to_name)
# metrics["per_class_ap50"] == {"player": ..., "ball": ..., "referee": ..., "rim": ..., "number": ...}
```

### The bootstrap report shape the accuracy CI table must read

```python
# Source: src/object_detection_eval/metrics/bootstrap.py::build_report (already committed,
# already validated byte-identical against the anchor in 04-03-SUMMARY.md)
# report["per_model"]["YOLO26m"]["mAP_50_95"] == {
#     "point_estimate": 0.7155195686816793,
#     "bootstrap_mean": 0.716498810032704,
#     "bootstrap_std": 0.00636155403724172,
#     "ci_2.5": 0.7040701945851645,
#     "ci_97.5": 0.7284898598808711,
# }
# report["pairwise"]["RTMDet-M minus DAMO-YOLO-M"]["mAP_50_95"]["ci_excludes_zero"] == False  # the tie
```

### Reading the latency honest-label

```python
# Source: benchmarks/basketball/results/latency/trt_fp16_toboxes.json (already committed, Phase 6)
import json
data = json.loads(Path("benchmarks/basketball/results/latency/trt_fp16_toboxes.json").read_text())
data["reproducibility"]["label"]
# == "manually measured 2026-07-21, not reproducible from this repo"
data["reproducibility"]["source_band_ms_2026_07_21"]  # == [4.0, 7.1] — the headline band to publish
data["reproducibility"]["second_run"]["outcome"]  # method-reproduces-but-not-portable explanation
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `EVAL_REPORT_FINAL.md`/`EVAL_REPORT.md` as hand-maintained markdown in the (soon-archived) training repo | Generated tables from committed JSON in `object-detection-eval`, with hand-authored prose around them | This phase (Phase 7) | The training repo's reports become historical references only; `object-detection-eval`'s reports are the ones a blog links to and the ones REPORT-01 makes tamper-evident. |
| `EVAL_REPORT_FINAL.md`'s prose claiming "every adjacent pair is significant" | Corrected: 5 of 6 significant, RTMDet-M vs DAMO-YOLO-M is a tie | Phase 4, `04-03-SUMMARY.md`, already landed in `docs/methodology.md` | Phase 7's reports must carry the corrected claim, not the original one. |
| §6 latency reported as a clean 4.0–7.1ms fp16 band | Explicitly labeled "manually measured 2026-07-21, not reproducible from this repo"; second-T4 run shows the method reproduces but the absolute band does not port | Phase 6, `06-03-SUMMARY.md` (LAT-04 honest-label resolution) | Phase 7's latency section must carry this caveat verbatim, not present the band as freshly reproduced. |

**Deprecated/outdated:**
- The training repo's `.deploy_comparison/`-relative reproduction path (`--source-repo`,
  `--yolox-root` CLI flags on `run_benchmark.py`) is a **developer/CI-mirror convenience**, not
  the public reproduction path. REPORT-05's public path is clone → registry weight fetch → ONNX
  end2end run — the source-repo flags should not appear in README's public instructions.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `report/` package name and file split (`loaders.py`/`tables.py`/`inject.py`) is a recommendation, not a locked decision — no CONTEXT.md exists yet for this phase to confirm it | Architecture Patterns / Recommended Project Structure | Low — purely an internal naming choice; the planner or `/gsd-discuss-phase` can adjust without invalidating any other finding. |
| A2 | Marker-comment injection (vs. e.g. a single fully-generated `<!-- GENERATED -->` file with prose in a separate include) is the right authoring model for mixing prose and tables in one `.md` | Architecture Patterns, Pattern 2 | Medium — if the planner prefers separate prose/data files (e.g. a `.md.j2` + rendered `.md`), the generator design changes, but the underlying data sources (results JSON, registry cards) and the REPORT-01 guarantee are unaffected. |
| A3 | `results/accuracy/*.json` should be produced via `--mode end2end` (registry/ONNX-driven, reproducible by an external reader) as the *canonical* committed artifact, with `--mode from-predictions` reserved for fast dev/CI-mirror verification against source-repo-only stored predictions | Summary, Architecture Patterns Pattern 1 | Medium — if end2end and from-predictions diverge beyond tolerance for some model on this specific write, the planner must decide which number is "the" published one; Phase 4's own numbers (04-01-SUMMARY.md) show both modes already agree to ≤0.005 for every model, so this risk is low but not zero. |
| A4 | A generator-side assertion that `registry/*.yaml` `evaluations` block values match the freshly-computed `results/accuracy/*.json` values is a "nice to have" cross-check, not a hard requirement | Architecture Patterns, Pattern 3 | Low — if skipped, no REPORT-01..05 requirement is unmet; it was proposed as an extra integrity check, not load-bearing. |

## Open Questions

1. **Exact CLI shape for `--write-results` on `run_benchmark.py`/`run_bootstrap_gate.py`**
   - What we know: both scripts already compute the full metrics dict / `build_report()` output
     in memory before printing; adding a write is mechanically simple (see Pattern 1's example).
   - What's unclear: whether `run_bootstrap_gate.py` should write ONE combined file (Check A's
     7-model anchor + Check B's YOLOX-M@800-vs-YOLO26m tie) or two separate files, and whether
     `run_benchmark.py`'s `--write-results` should accept a single path or infer the taxonomy
     suffix automatically from `--taxonomy`.
   - Recommendation: the planner should decide this as a concrete task-level design choice; either
     shape satisfies REPORT-01 as long as the resulting file(s) are committed and the generator
     reads them without re-deriving numbers.

2. **Does the 10-class per-class table belong in `FINAL_COMPARISON_640.md` at all, or only in an
   appendix/`docs/methodology.md`?**
   - What we know: `EVAL_REPORT_FINAL.md` §4 presents it as a secondary table ("DETR-family wins
     the 10-class task... on the coarse 5-class task, YOLO26m's strong ball/number recall puts it
     first") — a nuance, not the headline.
   - What's unclear: whether REPORT-02's "per-class AP" requirement is satisfied by 5-class alone
     (matching the "leading with the preprocessing finding... stating the tie up front" framing,
     which is entirely 5-class) or must include both.
   - Recommendation: include the 5-class per-class table in the main body (it directly supports
     the headline ranking) and the 10-class table either as a collapsed/appendix section or a
     one-paragraph callout with a link to the generated JSON — avoids diluting the lede while
     still satisfying "per-class AP" from the requirement text.

3. **Should `docs/methodology.md` and `README.md` also pull generated fragments (e.g. the 7-model
   table), or are they purely hand-written documents that merely reference the two reports?**
   - What we know: REPORT-04/05 describe narrative documentation (preprocessing methodology, the
     94-image statistical limitation, the clone→fetch→run→reproduce path) rather than data tables.
   - What's unclear: whether any generated table should be duplicated into `methodology.md` for
     convenience.
   - Recommendation: keep `methodology.md`/`README.md` as pure prose that links to
     `benchmarks/basketball/reports/*.md` for the generated tables, avoiding a second copy of any
     number that could drift — this is the simplest way to fully satisfy REPORT-01's "no
     published number can drift" guarantee (there is only ever one generated copy of each table).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `pixi` | running any script/test | ✓ | (on PATH; project memory notes `/Users/ortizeg/.pixi/bin/pixi` for reliability) | — |
| Core deps (`pydantic`, `loguru`, `pyyaml`, `supervision==0.29.1`) | report generator | ✓ | already locked in `pixi.lock` | — |
| `object-detection-training/.deploy_comparison/` (source_repo root) | `run_benchmark.py --mode from-predictions`, `run_bootstrap_gate.py` | ✓ present locally | — | Only needed for the from-predictions/CI-mirror path (A3); not needed for the public README reproduction path. |
| External `YOLOX/training_results` (yolox root) | `run_benchmark.py` (YOLOX-M ONNX/labels) | ✓ present locally | — | — |
| `basketball-player-detection-3/test/_annotations.coco.json` (test GT) | both accuracy scripts | ✓ present locally | — | — |
| Registry-published ONNX weights (HF Hub, via `download_weights()`) | the *public* end2end reproduction path (README's clone→fetch→run→reproduce) | Not verified in this session — `scripts/publish_weights.py` (REG-05) is a Phase 3 deliverable already marked complete; whether weights are actually uploaded to the HF Hub yet was not re-verified here | — | If not yet published, `run_benchmark.py --mode end2end` still works against the local ONNX paths used by Phase 4 (source_repo/yolox roots); the README's public-facing instructions should be validated against an actual `download_weights()` call before REPORT-05 ships. |

**Missing dependencies with no fallback:** none identified.

**Missing dependencies with fallback:** HF Hub weight availability (see table) — falls back to
the already-working local-path reproduction Phase 4 used, but REPORT-05's specific claim ("clone
→ fetch weights from the registry → run the benchmark → reproduce the table") should be smoke-
tested against a real `download_weights()` call during planning/execution, not assumed.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already configured, `[tool.pytest.ini_options]` in `pyproject.toml`) |
| Config file | `pyproject.toml` (`testpaths = ["tests"]`, `--cov-fail-under=80`) |
| Quick run command | `pixi run test` |
| Full suite command | `pixi run test-cov -m "not vlm and not trt and not external and not graphsurgeon"` (matches `.github/workflows/test.yml`'s exact CI invocation) |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REPORT-01 | Regenerating from a results-file fixture changes the rendered table; `--check` exits nonzero on drift | unit + CLI | `pixi run pytest tests/report/test_tables.py tests/scripts/test_generate_report.py -x` | ❌ Wave 0 (new) |
| REPORT-01 | A specific golden cell (e.g. YOLO26m 5c mAP@50:95 = 0.716) in the rendered `FINAL_COMPARISON_640.md` primary table equals `results/accuracy/reproduction_640_merged5.json`'s value | unit (golden-value) | `pixi run pytest tests/report/test_tables.py::test_primary_table_matches_results_file -x` | ❌ Wave 0 (new) |
| REPORT-02 | Rendered latency section surfaces the honest-label caption when `reproducibility.status == "manually_measured"` | unit | `pixi run pytest tests/report/test_tables.py::test_latency_honest_label -x` | ❌ Wave 0 (new) |
| REPORT-03 | Rendered VLM per-class table recomputes (not transcribes) AP50 from `results/vlm/*.json` via `compute_metrics` | unit (against a small synthetic GT/pred fixture, not the real 94-image set — keeps the test fast and hermetic) | `pixi run pytest tests/report/test_loaders.py::test_vlm_per_class_ap -x` | ❌ Wave 0 (new) |
| REPORT-01..03 | `run_benchmark.py --write-results` / `run_bootstrap_gate.py --write-results` produce valid JSON matching each new pydantic model | unit (offline, synthetic manifest — mirrors `tests/scripts/test_run_benchmark.py`'s existing pattern) | `pixi run pytest tests/scripts/test_run_benchmark.py tests/scripts/test_run_bootstrap_gate.py -x` | ✅ files exist; new test cases needed for the write path |
| REPORT-04/05 | No automated test — narrative documentation; verified by human review per `human_verify_mode: end-of-phase` (config.json) | manual-only | — | — (manual-only, justified: prose accuracy/tone is not mechanically checkable) |

### Sampling Rate

- **Per task commit:** `pixi run test` (fast subset touching `report/` + the two script
  extensions)
- **Per wave merge:** `pixi run test-cov -m "not vlm and not trt and not external and not
  graphsurgeon"` (matches CI exactly)
- **Phase gate:** Full suite green before `/gsd-verify-work`, plus a manual read-through of both
  generated reports and `docs/methodology.md`/`README.md` (REPORT-04/05 are manual-only per the
  table above)

### Wave 0 Gaps

- [ ] `tests/report/__init__.py`, `tests/report/test_loaders.py`, `tests/report/test_tables.py`,
      `tests/report/test_inject.py` — new package, covers REPORT-01/02/03
- [ ] `tests/report/fixtures/` — small synthetic results JSON (accuracy, bootstrap, latency, VLM
      shapes) so table-rendering tests don't depend on the real 94-image committed data (keeps
      tests fast, hermetic, and independent of future results-file regeneration)
- [ ] `tests/scripts/test_generate_report.py` — CLI-level `--check`/`--write` behavior, offline
- [ ] Extend `tests/scripts/test_run_benchmark.py` and `tests/scripts/test_run_bootstrap_gate.py`
      with cases for the new `--write-results` flag (offline, synthetic manifest, no external
      artifacts — mirrors the existing test pattern in both files)
- Framework install: none — pytest and all fixtures needed are already available; no new
  framework or plugin required.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A — no auth surface in this phase (no new API, no new credential use). |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | yes | `report/loaders.py` pydantic models validate every results-file field it reads (mirrors `ManifestEntry`/`ModelCard`'s existing validation pattern); `inject.py`'s marker regex validates exactly-one-match before replacing (raises, does not silently no-op or double-replace). |
| V6 Cryptography | no | N/A — no crypto in this phase (weight SHA-256 verification is already REG-03's concern, unaffected here). |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A malformed/hand-edited `results/*.json` silently producing a wrong or missing table cell | Tampering | pydantic model validation on load (`extra="forbid"` where the shape is fully known, e.g. `AccuracyResult`/`BootstrapReport`) so a malformed or unexpectedly-shaped file fails loudly at generation time rather than rendering a blank/wrong cell. |
| Marker-injection regex matching the wrong section (or zero/multiple sections) and corrupting hand-written prose | Tampering / Denial of Service (of the doc) | `inject_table()` asserts exactly one marker-pair match and raises `ValueError` otherwise (see Pattern 2's code example) — never silently no-ops or applies to an unintended span. |
| A future contributor adding a new results field that the generator silently ignores, causing a report to look complete while missing new data | Tampering (of trust, not of bytes) | Frozen, `extra="forbid"`-style pydantic models (matching `ModelCard`'s existing convention) make an unrecognized field a load-time error, not a silent drop. |

## Sources

### Primary (HIGH confidence) — all `[VERIFIED: local codebase]`, read directly this session

- `object-detection-eval/scripts/run_benchmark.py` (read in full) — REPRO-01 gate structure, no
  existing write path
- `object-detection-eval/scripts/run_bootstrap_gate.py` (read in full) — REPRO-03 gate structure,
  `build_report()` shape, Check A/B logic
- `object-detection-eval/src/object_detection_eval/metrics/bootstrap.py` (read in full) —
  `load_predictions`, `run_bootstrap`, `build_report` public API
- `object-detection-eval/src/object_detection_eval/metrics/detection_map.py` (read in full) —
  `compute_metrics` return shape including `per_class_ap50`
- `object-detection-eval/benchmarks/basketball/conf/{reproduction_640,vlm_zeroshot,latency_640}.yaml`
  (read in full) — manifest shapes, expected values, per-model protocol notes
- `object-detection-eval/benchmarks/basketball/results/{latency/*.json,vlm/*.json}` (inspected
  via `python3 -c` and `wc -l`) — actual committed data shapes, reproducibility field, git-tracked
  status confirmed via `git ls-files`/`git check-ignore`
- `object-detection-eval/registry/yolo26m_640.yaml` (read) — `ModelCard` shape, `evaluations`
  block
- `object-detection-eval/src/object_detection_eval/registry/download.py` (read, partial) —
  `ChecksumMismatchError`/`WeightsNotRedistributableError` public API for README's reproduction
  path
- `object-detection-eval/.planning/{ROADMAP.md,REQUIREMENTS.md,config.json,STATE.md}` (read in
  full) — phase goal, requirement text, workflow config (`nyquist_validation: true`,
  `security_enforcement: true`), current phase status
- `object-detection-eval/.planning/phases/{04-reproduction-gate,05-zero-shot-vlm,06-latency}/*-SUMMARY.md`
  (read in full) — exact reproduced numbers (per-model CIs, pairwise significance, VLM box-run
  numbers, latency honest-label second-run outcome)
- `object-detection-eval/docs/{methodology.md,FORK_PLAN.md}`, `README.md` (read in full) —
  existing corrected drift narrative, blog-post mapping (§12), current README state ("Status:
  scaffold" — stale, REPORT-05 must update it)
- `object-detection-training/eval_output/EVAL_REPORT_FINAL.md` (read in full, 427 lines) — the
  authoritative source content for REPORT-02 (§2 primary table + CIs, §3/§4 per-class AP,
  §5a fairness audit, §6 latency)
- `object-detection-training/eval_output/EVAL_REPORT.md` (grepped + read lines 410-530) — legacy/
  superseded VLM per-class AP table (used only as a cross-check reference, not a source of
  numbers per Anti-Patterns)
- `object-detection-training/.deploy_comparison/bootstrap_5c_test_7models.json` (inspected via
  `python3 -c`) — confirmed the anchor JSON's shape is byte-identical to `build_report()`'s output
  shape

### Secondary (MEDIUM confidence)

None — every claim in this document traces to a primary local-codebase source; no external
documentation lookup was needed since Phase 7 introduces no new library or external API.

### Tertiary (LOW confidence)

None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependency; every library cited is already installed, pinned,
  and in active use elsewhere in this exact codebase.
- Architecture: HIGH — the `report/` package slot was explicitly anticipated in `ROADMAP.md`'s
  own Phase 2 description; the executor-run/commit-then-read pattern is directly copied from two
  already-shipped phases (5 and 6) in this same repo.
- Pitfalls: HIGH — all four pitfalls are drawn from already-documented, already-resolved
  incidents in this repo's own `docs/methodology.md` and Phase 4/6 summaries (the RTMDet-M/
  DAMO-YOLO-M tie correction, the LAT-04 honest-label resolution), not speculative.

**Research date:** 2026-07-29
**Valid until:** No expiry driver — this is an internal-architecture research document tied to
this specific repo's state as of 2026-07-29 (Phase 6 complete), not to any external library
version. Re-research only if `compute_metrics`/`build_report`'s public signatures change, or if
Phases 1-6's committed results files are regenerated with a different shape.
