# Cost per frame of the zero-shot API pipelines (2026-10)

Plan of record, written before any rerun result exists. The rules below are fixed now;
a number that comes out of the reruns can change a table, never a rule.

## Question

The blog's claim is that you can rent a zero-shot detector. What does each accuracy level
cost a team, in dollars and in seconds, per frame? The accuracy half is already published
(`benchmarks/basketball/reports/VLM_VS_FINETUNED.md`). This study measures the other half
and joins the two in `benchmarks/basketball/reports/COST.md`.

Every number must be reproducible from this repo: cost comes from committed per-call usage
logs, a committed price table with sources, and a generator. Nothing is hand arithmetic.

## What is measured

Eight pipelines, each at its **published config and its published `--workers`**, and two
fused lines that are sums of their members.

| id | published config (source of truth) | env | workers | evidence |
| --- | --- | --- | --- | --- |
| `astra` | `vlm_zeroshot.yaml` row `astra` (gpt-6-astra, effort low) | `vlm-astra` | 1 | `run_vlm_benchmark.py` had no `--workers` at 0ed62dd; at HEAD only Gemini rows are concurrent |
| `astra_crop_refine` | `run_astra_crop_refine.py --model astra` defaults (`_BASE_PROMPT`, effort low, pad 1.5, refine ball,rim) | `vlm-astra` | 1 | the script has no `--workers`; `score_split` defaults to 1 |
| `sol` | `vlm_zeroshot.yaml` row `sol` (gpt-6.1-sol, effort default) | `vlm-astra` | 1 | no `--workers` at 909bcb1 |
| `claude_opus` | `vlm_zeroshot.yaml` row `claude_opus` (claude-opus-5-5, effort default) | `vlm-claude` | 1 | no `--workers` at 909bcb1 |
| `gemini_argon_single` | `vlm_ablation.yaml` arm `rim_def_argon` (= the Oct-14 test row): gemini-4-argon, `box_2d`, rim line, timeout 300, no thinking, no pipeline | `vlm` | 1 | **INFERRED, not recorded.** `run_vlm_benchmark.py` had no `--workers` until 9c46399 (2026-10-21), so the Oct-14 test run was serial; the val arms used `ablate_vlm.py --workers N` with N unrecorded (capped at 3) |
| `gemini_argon_two_pass` | `vlm_zeroshot.yaml` row `gemini` (thinking high, `crop_refine+number_crop`) | `vlm` | 3 | `gemini-argon-2026-10.md` "Round 3 test": "With `--workers 3`, the repeat took 95 minutes" |
| `dinox_pro_rimtrim` | `dinox_explore.yaml` arm `r6_trim0.35_headline` (DINO-X-1.0, 2x2 tiles + full frame, merge NMS 0.5, trim 0.35) | `default` | 8 | **ASSUMED default.** `explore_dinox.py --workers` defaults to 8 at ff353c8 and the docs record no override |
| `dinox_preview_rimtrim` | `dinox_explore.yaml` arm `r9_trim0.3_preview` (DINO-X-PREVIEW-1.0, same tiling, trim 0.3) | `default` | 8 | **ASSUMED default**, as above |

Derived lines, with zero calls of their own (WBF IoU 0.55 is free):

- `fusion4` = `astra_crop_refine` + `dinox_pro_rimtrim` + `dinox_preview_rimtrim` +
  `gemini_argon_two_pass` (the best zero-shot line, 0.683).
- `api_fusion4` = `astra` + `sol` + `claude_opus` + `gemini_argon_two_pass` (the API-rows
  fusion, 0.598). Every member is already rerun, so this line is free.

The driver builds each pipeline from the code that produced its published row
(`run_vlm_benchmark.py` factories, `ablate_vlm.py` builders, an extracted
`build_crop_refine_pipeline` in `run_astra_crop_refine.py`, `explore_dinox.py::_build`), so no
published configuration is re-typed. The extraction is a pure refactor. Config drift since
the published crop-refine run (0ed62dd, 2026-09): none. The pass-1 prompt, effort (low), pad
(1.5), refine classes (ball, rim) and both refiner prompts are byte-identical at HEAD.
`conf/cost_rerun.yaml` pins what each source must resolve to (`expect`), and
`scripts/cost_rerun.py check-config` fails if any source has drifted.

## The frame rule

20 validation frames, committed in `benchmarks/basketball/conf/cost_val_frames.json`.

- Seed `20261027`; stratified by clip with a largest-remainder allocation. The three val
  clips have 31 / 32 / 33 frames, so 20 splits 6 / 7 / 7.
- Ordered round-robin across clips, so any prefix spans every clip: the 3-frame projection
  run uses the first 3 frames, one per clip.
- Only filenames are committed. `scripts/cost_rerun.py select-frames --check` reproduces
  the file from the local dataset.
- All frames are 1920x1080 on every split, so image-token counts measured on val transfer
  to the test geometry.

## The run rule

- **Validation split only. The test split is never touched.** The driver has no split option;
  it refuses any frame that is not in `valid/_annotations.coco.json` or that appears in
  `test/_annotations.coco.json`.
- **Two runs** per pipeline, labelled `1` and `2`.
- **Every cache and checkpoint is bypassed.** The driver calls `predict` once per frame
  itself, never `score_split` (which has a checkpoint) or `load_or_collect_raw` (the
  ablation raw cache). The DINO-X arms are rebuilt with `no_cache=True` and `cache_dir=None`,
  so every frame is a billed call.
- **The published `--workers`**, from the config; there is no override.
- **Abort on the first failure.** An error usage record or an exception stops frames not yet
  started; the run is written as `aborted` and is never averaged in. Rerun it with `--force`.
- **Sequential across pipelines**, so latency is not distorted by another pipeline's load.
- Each run writes one usage JSONL and one run manifest (git sha, pixi env, SDK versions,
  workers with citation, frames, per-frame timings, frames-file sha256).

## Metric definitions

- **Calls/frame**: every billed request for the frame, including second-pass crop calls and
  tile calls; frames with no calls count as 0. Reported as mean and max, and by pass.
- **Tokens/frame**: input (with the image / text split where the API reports it, and the cached
  part), billable output, and reasoning.
- **USD/frame**: run cost divided by the run's frame count; mean and SD (ddof = 1) over the 2
  runs. **USD per 1,000 frames** = mean x 1000. With 2 runs the SD is rough.
- **Record cost** = input tokens x input rate + billable output x output rate, per 1M tokens.
  Billable output is `output_tokens` when the provider already counts reasoning inside it
  (OpenAI, Anthropic) and `output_tokens + reasoning_tokens` when it does not (Gemini).
  Cached input is billed at the full input rate and flagged. A DINO-X call costs the per-call
  price. **A null price is unpriced, never zero.**
- **Serial s/frame**: per-run mean of each frame's own wall time, then mean and SD over runs.
  **Concurrent s/frame**: run wall time / frames, which is what the published concurrency
  delivers.
- **Fused line** = the sum of its members, run index by run index (run k = sum of members'
  run k). Its calls/frame mean is the sum of member means and its max the sum of member
  maxima (an upper bound, labelled as such). Its latency is the sum of members (they run one
  after another), with the slowest member reported as the lower bound if they ran in parallel.
  A fused line with an unpriced member has no USD figure, only a lower bound that leaves the
  unpriced members out.
- **Accuracy** is copied from the committed reports, each value with its citation
  (`conf/cost_published_accuracy.yaml`), and a test re-verifies every citation against the real
  file. It is never recomputed here.

## Price scenarios

`conf/cost_prices.yaml` holds per-model prices with effective dates and source URLs, as
supplied by the user and cross-checked on 2026-10-27. Two scenarios are reported:

- **current**: each record priced at the price in force on its own date.
- **post_2026_12_01**: every record priced as of 2026-12-01. Only Gemini differs: the
  `gemini-4-argon` price steps from $2 / $10 to $4 / $20 per 1M tokens (input / output).

| model | input / output per 1M tokens | source |
| --- | --- | --- |
| gpt-6-astra | $10 / $50 | developers.openai.com/api/docs/pricing |
| gpt-6.1-sol | $2 / $10 | developers.openai.com/api/docs/pricing |
| claude-opus-5-5 | $4 / $20 | claude.com/pricing |
| gemini-4-argon | $2 / $10 through 2026-11-30, then $4 / $20 | ai.google.dev/gemini-api/docs/pricing |
| DINO-X-1.0, DINO-X-PREVIEW-1.0 | per-call price **not yet supplied** | DeepDataSpace console |

## The budget rule

- Cap: **40 USD** for the whole study.
- Before any full run, each pipeline runs its first 3 frames (`--run proj --max-frames 3`).
  `cost_rerun.py project` extrapolates: spent so far (the projection records themselves) +
  mean USD/frame x 20 frames x 2 runs, summed over priced pipelines. **Projection calls count
  toward the cap.**
- If the projection exceeds the cap, `project` exits 3. **Stop and ask the user before any
  further paid call.**
- DINO-X is unpriced until the user supplies a per-call price, so it is excluded from the USD
  projection and shown as projected call counts. The user is told this.

## Step 2 scope

By user decision on 2026-10-27, the OpenAI and Anthropic admin billing checks were skipped:
no admin-key reader, no billing-comparison column in `COST.md`, no 25 % flag. What remains is
two read-only instruction pages for the user:

- `docs/cost/gcp-billing-export.md`: Google Cloud Billing report by SKU and day for the
  Gemini key's project, 2026-10-01 to the rerun date.
- `docs/cost/dinox-console-checklist.md`: DeepDataSpace console: the per-call price, balance
  before and after, daily consumption against the repo's documented call counts, recharges.

The DINO-X per-call price is pending from the user.

## Tooling (built, zero API calls)

- `inference/vlm/usage.py`: a permanent usage log. Every billed request of Astra, Sol, Opus,
  Gemini, DINO-X and T-Rex appends one JSONL record (default sink
  `~/.cache/object-detection-eval/api_usage.jsonl`, `ODE_API_USAGE_LOG` overrides, `off`
  disables). Predictions and DINO-X cache keys are unchanged (golden tests).
- `scripts/cost_rerun.py`: `select-frames`, `check-config`, `run`, `project`.
- `scripts/cost_report.py`: usage logs + run manifests -> `cost_per_frame.json` -> `COST.md`
  + `cost_vs_accuracy.png`, with a `--check` drift gate.

## Command sequence

Run from the repo root, one pipeline at a time. Verify real artifacts after each command
(manifest `status` is `complete`, usage line counts), not a background-task notification.
Keys come from the environment: `OPENAI_KEY` or `OPENAI_API_KEY`; `ANTHROPIC_KEY` or
`ANTHROPIC_API_KEY`; `GEMINI_API_KEY` or `GOOGLE_API_KEY`; `DINOX_KEY` or `DDS_API_TOKEN`.

1. Offline checks (free):

   ```
   pixi run python scripts/cost_rerun.py check-config
   pixi run python scripts/cost_rerun.py select-frames --check
   ```

2. Projection, 3 frames each:

   ```
   pixi run -e vlm-astra python scripts/cost_rerun.py run --pipeline astra --run proj --max-frames 3
   pixi run -e vlm-astra python scripts/cost_rerun.py run --pipeline astra_crop_refine --run proj --max-frames 3
   pixi run -e vlm-astra python scripts/cost_rerun.py run --pipeline sol --run proj --max-frames 3
   pixi run -e vlm-claude python scripts/cost_rerun.py run --pipeline claude_opus --run proj --max-frames 3
   GEMINI_API_KEY="$GEMINI_KEY" pixi run -e vlm python scripts/cost_rerun.py run --pipeline gemini_argon_single --run proj --max-frames 3
   GEMINI_API_KEY="$GEMINI_KEY" pixi run -e vlm python scripts/cost_rerun.py run --pipeline gemini_argon_two_pass --run proj --max-frames 3
   pixi run python scripts/cost_rerun.py run --pipeline dinox_pro_rimtrim --run proj --max-frames 3
   pixi run python scripts/cost_rerun.py run --pipeline dinox_preview_rimtrim --run proj --max-frames 3
   pixi run python scripts/cost_rerun.py project
   ```

   Exit 3 means the projection exceeds 40 USD: stop and ask the user.

3. Full runs, `--run 1` then `--run 2` for each of the 8 pipelines, same environments and
   key mapping as step 2, without `--max-frames`:

   ```
   pixi run -e vlm-astra python scripts/cost_rerun.py run --pipeline astra --run 1
   ```

   Exit 4 means a call failed: inspect, then rerun that (pipeline, run) with `--force`.

4. Generate and check the report:

   ```
   pixi run python scripts/cost_report.py
   pixi run python scripts/cost_report.py --check
   ```

5. Supply the DINO-X per-call price (`docs/cost/dinox-console-checklist.md`) in
   `conf/cost_prices.yaml`, then repeat step 4.
6. The user performs the Google export and the DINO-X console checklist.
7. Draft, but do not commit into the report, a proposed `VLM_VS_FINETUNED.md` paragraph from
   `COST.md`. That report is not edited by this work.
8. Commit the usage logs, run manifests, `projection.json`, `cost_per_frame.json`, `COST.md`
   and the PNG on a branch, staging explicit paths.

## Status

- [x] Per-call usage logging in every API inferencer (golden cache-key tests)
- [x] Price table and published-accuracy config, with citations verified by test
- [x] Aggregator, `COST.md` and PNG generator, `--check` gate
- [x] 20-frame validation list committed
- [x] Rerun driver and projection gate
- [x] Step 2 docs (Google export, DINO-X checklist)
- [ ] Projection runs (3 frames x 8 pipelines)
- [ ] Projection within the 40 USD cap (or user decision)
- [ ] Full runs 1 and 2 of 8 pipelines
- [ ] `COST.md` generated and `--check` clean
- [ ] DINO-X per-call price supplied
- [ ] Google export and DINO-X console checklist done by the user
