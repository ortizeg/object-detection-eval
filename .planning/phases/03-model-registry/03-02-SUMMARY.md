---
phase: 03-model-registry
plan: 02
status: complete
requirements: [REG-03, REG-04]
---

# 03-02 Summary — Verified download + AGPL guard

Both tasks complete. Full suite green (152 passed, 95.9% coverage), ruff (T20) +
mypy strict clean, torch-free.

## Task 1 — download.py (REG-03) (commits `8082697` RED, GREEN below)
Ported the model-zoo download tier into `registry/download.py`: `sha256_file`,
`verify_file`, `cached_path`, `default_fetcher` (file/http scheme dispatch,
rejects unknown schemes), and `download_weights(card, *, cache_dir, fetcher,
force)` — streams to `.part`, hashes on arrival, promotes on match, raises
`ChecksumMismatchError` (leaving no `.part` or cache file behind), and re-fetches
a corrupt cache entry. All tested with `file://` fixtures + injected fetchers, no
network.

## Task 2 — REG-04 guard (safety-critical)
`WeightsNotRedistributableError` is raised as the **first statement** of
`download_weights` — before `cached_path`, before any fetcher call — for a
`redistributable: false` card OR a card with no `weights`. The message names the
card key + its reproduction command/source. Proven guard-first by an
exploding-fetcher test asserting `cache_dir` is never even created.

## Deviation
- Task 1's GREEN `download.py` (and the `registry/__init__.py` export update) was
  authored by the orchestrator after the executor subagent hit an API
  "connection closed" error immediately after committing the RED tests (`8082697`).
  Same-file scope; verified against the committed RED suite (12 tests) + full suite.

## Requirements
- REG-03 ✅ · REG-04 ✅ (guard-first, named error, no I/O before the guard)
