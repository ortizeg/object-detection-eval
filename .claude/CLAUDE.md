<!-- GSD:project-start source:PROJECT.md -->

## Project

**object-detection-eval**

A reproducible evaluation harness for object detection networks, applied to a deliberately
small dataset (basketball: 464 train / 96 val / 94 test images). It scores fine-tuned
detectors and zero-shot VLMs under one identical protocol, and it is the public code
companion to two blog posts. It consumes trained models as artifacts — it does not train
anything.

**Core Value:** **Every number the blog posts publish must be reproducible from this repo.** If a reader
clones it, fetches the weights, and runs the harness, they get the numbers in the reports.
Everything else — genericity, VLM coverage, latency tables — is secondary to that.

### Constraints

- **Tech stack**: pixi for environments, never pip/conda. Hydra for config, Pydantic v2 for
  schemas, loguru for logging (never `print()` — ruff `T20` enforces it).

- **Dependencies**: the core package must import without torch. Torch lives behind `[vlm]`,
  TensorRT behind `[trt]`. This keeps CI fast, macOS-clean, and the reader install small.

- **Statistical honesty**: 94 test images. YOLOX-M vs YOLO26m is a statistical tie
  (+0.73 pt, CI [−0.33, +1.90]). Reports must lead with this, not bury it.

- **Licensing**: repo is Apache-2.0. Evaluated models carry their own licenses; AGPL weights
  are not redistributed.

- **Dataset**: CC BY 4.0, from `ego-playground/basketball-player-detection-3` on Roboflow.
  Redistributable with attribution.

- **Hardware**: Phase 5 latency work needs a T4; the original instance is gone. Budget a few
  vast.ai GPU-hours.

- **Storage**: private source of truth stays at
  `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/` (1.89 GiB verified).
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->

## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
