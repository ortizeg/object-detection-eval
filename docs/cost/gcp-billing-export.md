# Google Cloud Billing export for the Gemini key (by SKU and day)

> The OpenAI and Anthropic admin billing checks were skipped by user decision on
> 2026-10-27, so there is no billing-comparison column in `COST.md`. This page is
> the Google half of the cross-check the study can still offer: it is read-only,
> the user performs it in the console, and nothing in this repo calls a billing
> admin API.

## What this is for

`COST.md` prices every Gemini call from its own usage record (token counts from the
API response times the price table). The Google Cloud Billing report is an
independent total. If the two agree to within rounding, the token-to-USD arithmetic
is right. If they differ, the likely causes are listed at the end.

The Gemini key ran on **prepaid credits** during the Argon work (see
`docs/plans/gemini-argon-2026-10.md`, "Operational": the credits ran out at 19:40 on
2026-10-15 and were topped up). Prepaid credits can show up as a credit line rather
than as a charge, so check both views below.

## Window to export

2026-10-01 through the date of the logged rerun. That covers:

- 2026-10-12: the `gemini-3.8-flash` refresh and the ten parallel val sweeps;
- 2026-10-14 to 2026-10-21: the Gemini 4 Argon work (rim line, thinking, crop and
  number-crop pipelines, the repeat test run at `--workers 3`);
- the cost-per-frame rerun itself (2 runs x 2 Gemini pipelines + the 3-frame
  projections).

## Steps: Cloud Billing report (no setup, works for past days)

Source: <https://cloud.google.com/billing/docs/how-to/reports> (fetched 2026-10-27).

1. Open the Google Cloud console, then **Billing** and choose the billing account
   that owns the project the Gemini API key belongs to. (If the key was created in
   Google AI Studio, the AI Studio project is listed under that account.)
2. Open **Reports**.
3. **Time range**: choose a custom range, 2026-10-01 to the rerun date.
4. **Group by**: choose **SKU**. Optionally set the report to show daily cost, so the
   chart has one bar per day (the page's default view is daily cost).
5. **Filters**: under **Services** select *Generative Language API* (the service the
   Gemini API is billed under; if the list shows a different Gemini-related service
   name, select that one). Leave *Include credits* as it is, then repeat once with
   credits excluded so the gross usage cost is visible beside the net.
6. Click **Download CSV** (the selector above the summary table). The download is
   limited by the filters set and includes every row and column of the table.
7. Save the CSV and note, for each SKU, the cost for each of 2026-10-12,
   2026-10-14 through 2026-10-21, and the rerun date.

Read off the SKUs named for the Gemini models you used (`gemini-3.8-flash` before the
Argon work, `gemini-4-argon` after). Thinking tokens are billed at the output rate
(<https://ai.google.dev/gemini-api/docs/pricing>: "Output price (including thinking
tokens)"), so they are not a separate line to add.

## Steps: BigQuery billing export (only if already enabled)

Source: <https://cloud.google.com/billing/docs/how-to/export-data-bigquery> (fetched
2026-10-27).

Check **Billing export** in the console. If standard usage cost export to BigQuery is
already enabled, query it for the same window grouped by `sku.description` and
`DATE(usage_start_time)`; the table gives the same totals with per-day, per-SKU rows
and the credit lines.

Do not enable it just for this: per the docs, a multi-region dataset back-fills only
from the start of the previous month, and a regional dataset is **not** back-filled at
all, so enabling it now would not recover the Argon-era days. The Reports page above
covers those days without any setup.

## AI Studio usage view

Because the key ran on prepaid credits, also open the AI Studio usage page for the
project and note the requests and tokens per model per day for the same window. It is
the closest thing to a call count on Google's side and can be compared with the number
of Gemini records in `results/cost/usage/` for the rerun date.

## What to send back

- the SKU-by-day CSV (or the totals per SKU for the rerun date and for the
  2026-10-12 and 2026-10-14..21 windows);
- the credit balance before and after the rerun, if shown;
- the AI Studio request counts for the rerun date.

Then compare, for the rerun date only, the report's gemini-4-argon cost with
`study_spend_by_file` for the two Gemini pipelines in `cost_per_frame.json`.

## If the numbers do not match

- Prepaid credits can make the net charge zero. Compare the **gross** usage cost.
- The Gemini 4 Argon price steps on 2026-12-01 (input $2 to $4, output $10 to $20 per
  1M tokens). A rerun before that date is priced at the first price in both places.
- Retried requests that failed server-side may or may not be billed; the usage log
  records one line per logical request and counts the retries in `retries`.
- The report is in the account's currency and the console may show a different tax
  treatment than the pre-tax usage cost the study computes.
