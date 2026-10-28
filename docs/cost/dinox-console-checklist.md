# DINO-X (DeepDataSpace) console checklist

> The OpenAI and Anthropic admin billing checks were skipped by user decision on
> 2026-10-27, so there is no billing-comparison column in `COST.md`. DINO-X is the
> one provider whose price the repo cannot know: the per-call price is a console
> fact. Until the user fills it in, every DINO-X pipeline, and every fused line
> containing one, is reported **unpriced** (never as zero). This checklist is for the
> user to complete in the DeepDataSpace console, read-only.

Console: <https://cloud.deepdataspace.com/dashboard/usage> (the page the vendor's
README links for purchasing and viewing API calls:
<https://github.com/IDEA-Research/DINO-X-API>).

The task API itself documents no cost or consumption field on its trigger or
task-status responses (checked 2026-10-27 against the DINO-X-API README and the
open-source `dds-cloudapi-sdk`), so the usage log records the `task_uuid` and the poll
count but no cost. That is why the numbers below have to come from the console.

## 1. Per-call price (the blocking item)

**Filled in 2026-10-28.** The user supplied about CNY 0.15 per direct model API call,
applied to both DINO-X models and converted at 1 USD = 6.6684 CNY (2026-10-27,
<https://www.valutafx.com/history/usd-cny-2026-10-27>). It is approximate and was not
cross-checked against the console balance; sections 2–4 remain open for that.

Copy, for each model, the price of one API call, and the unit the console uses
(per call, per image, or per token package):

| Model id | Used by | Per-call price (USD) |
| --- | --- | --- |
| `DINO-X-1.0` | `dinox_pro_rimtrim` (full frame + 4 tiles = 5 calls per frame) | 0.0225 (about CNY 0.15) |
| `DINO-X-PREVIEW-1.0` | `dinox_preview_rimtrim` (5 calls per frame) | 0.0225 (about CNY 0.15) |
| `T-Rex-2.0` | few-shot only; not in this study, listed for completeness | |

If the console sells call packages, divide the package price by the number of calls it
buys and write down which package you used. If the unit is not "per call" (for
example, a quota that depends on image size), write the rule down instead of a number
and ask before filling in `cost_prices.yaml`.

## 2. Balance before and after the rerun

- Balance (calls or currency) **before** the first rerun command: ______
- Balance **after** the last rerun command: ______
- Calls the repo logged for DINO-X in the rerun (sum of the `deepdataspace` lines in
  `results/cost/usage/dinox_*__*.jsonl`): ______

The difference in balance divided by the logged call count is an independent check of
the per-call price.

## 3. Consumption by day

Copy the daily consumption for each of these days (calls and, if shown, currency):

- 2026-10-11, 2026-10-12, 2026-10-13, 2026-10-14, 2026-10-15
- the rerun date

Compare the call counts with the ones this repo documents, from
`docs/plans/dinox-pro-2026-10.md`:

| Day | Repo's documented DDS calls |
| --- | --- |
| 2026-10-13 | 1,340 ("Calls spent on 2026-10-13": 12 + 470 + 576 + 282) |
| 2026-10-15 | 1,061 ("Calls spent on 2026-10-15": 5 probes + 480 + 576) |

A match confirms that one logical request is one billed call (so the usage log's
"one record per request, retries counted in `retries`" is the right unit). A console
count above the repo's suggests retried or failed tasks are billed too; send both
numbers back.

## 4. Recharge history

Copy every recharge (date, amount, number of calls bought). An out-of-quota event
(HTTP 429, application code 429001, "Token out of Quota") was observed on 2026-10-11,
per `inference/vlm/dinox.py`; the recharge history shows when it was cured.

## 5. Where the numbers go

1. Put the per-call price in `benchmarks/basketball/conf/cost_prices.yaml`, as
   `per_call_usd` under `DINO-X-1.0` and `DINO-X-PREVIEW-1.0` (and `T-Rex-2.0` if you
   want it), keeping the `source_url` and replacing the `note`.
2. Regenerate the report, then check it:

   ```
   pixi run python scripts/cost_report.py
   pixi run python scripts/cost_report.py --check
   ```

3. The DINO-X rows, the fused line that contains them (`fusion4`), the PNG and the
   study spend then fill in; the "lower bound" figures disappear.
