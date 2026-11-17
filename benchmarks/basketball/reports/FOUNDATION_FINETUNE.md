# Fine-Tuning Foundation Models on 464 Images

[VLM_VS_FINETUNED.md](VLM_VS_FINETUNED.md) scored open-weights foundation models
zero-shot. This report fine-tunes the same models, and a few more, on the 464
training images and asks: **with light fine-tuning, do they beat the conventional
detectors trained from COCO weights in
[FINAL_COMPARISON_640.md](FINAL_COMPARISON_640.md)?**

Every arm was selected on `valid` and predicted on `test` once. The prediction dumps
are committed under `results/finetune/` and scored here with the same scorer as every
published row (`score_vlm_metrics`). Every numeric table below is emitted by
`scripts/generate_report.py` from `results/finetune/finetune_metrics_merged5.json` and
the two `results/bootstrap/*finetune.json` files, and injected between
`<!-- TABLE:... -->` markers. `--check` fails if a table drifts from those files.

## Read this first: three clips is very little evidence

<!-- TABLE:ft_caveat START -->
**94 test images = 3 video clips (30 / 33 / 31 frames).** The clip-clustered bootstrap enumerates all 10 distinct clip resamples.

- Of 26 main arms (peft and full, seed 0): **7 tie YOLO26m**, 7 above, 12 below (clustered 95% CI). The image-level bootstrap would call 24 of 26 significant.
- Of 35 all scored rows (seeds and variants included): **9 tie YOLO26m**, 10 above, 16 below (clustered 95% CI). The image-level bootstrap would call 33 of 35 significant.

**What "above" means with 3 clips.** 7 of the 7 main-arm rows called above YOLO26m won all 3 clips. With 3 clusters the percentile interval excludes zero when a row wins every clip by a margin no resample can reverse, and a one-sided sign test on 3 of 3 clips gives p = 1/8 = 0.125. So "above" reads as "ahead on every clip we have", not as significance at 5%; only more test clips can tighten it.
<!-- TABLE:ft_caveat END -->

The test split is three short video clips, and frames from one possession are near
duplicates. The clip-clustered bootstrap resamples clips, not frames, so its intervals
are wide and close rows are ties: the interval includes zero. "Tie" does not mean the
two models are equal; it means this test set cannot tell them apart.

The rows the clustered interval does separate from YOLO26m are the ones ahead on all
three clips. That is a consistent result and a weak one: three clips from three games
are three observations. Read "above" as "worth confirming on more footage", not as an
established ranking. The image-level bootstrap, which treats the 94 frames as
independent, is shown further down for contrast only; its intervals are too narrow
and its "significant" calls should not be quoted.

Every verdict below is against YOLO26m, the best published detector. Against DEIM-M
see the contrast table.

## Results by model

<!-- TABLE:ft_main START -->
| Model | Arm | Trainable | Val | Test | Δ vs YOLO26m [clustered 95% CI] | Verdict | Zero-shot → fine-tuned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLMDet-large | peft | 11.4% (39.0M / 343.1M) | 0.739 | 0.776 | +0.060 [+0.036, +0.102] | above | 0.388 → 0.776 (+0.387, test) |
| LLMDet-large | full | 100.0% (343.1M / 343.1M) | 0.739 | 0.785 | +0.070 [+0.046, +0.100] | above | 0.388 → 0.785 (+0.397, test) |
| MM-Grounding-DINO-B | peft | 16.0% (37.2M / 233.0M) | 0.713 | 0.749 | +0.034 [+0.007, +0.089] | above | 0.396 → 0.713 (+0.317, val) |
| MM-Grounding-DINO-B | full | 100.0% (233.0M / 233.0M) | 0.713 | 0.744 | +0.029 [+0.002, +0.082] | above | 0.396 → 0.713 (+0.317, val) |
| SAM 3 | peft | 4.6% (38.5M / 838.2M) | 0.668 | 0.702 | -0.014 [-0.064, +0.048] | tie | 0.479 → 0.668 (+0.189, val) |
| SAM 3 | full | 100.0% (838.2M / 838.2M) | 0.709 | 0.737 | +0.021 [-0.019, +0.062] | tie | 0.479 → 0.709 (+0.229, val) |
| YOLOE-26x @1280 | peft (`yoloe_peft`) | 0.02% (11.6K / 58.8M) | 0.400 | 0.414 | -0.301 [-0.345, -0.228] | below | 0.278 → 0.414 (+0.136, test) |
| YOLOE-26x @1280 | full (`yoloe_full`) | 100.0% (58.8M / 58.8M) | 0.774 | 0.775 | +0.060 [+0.037, +0.088] | above | 0.278 → 0.775 (+0.498, test) |
| YOLOE-26x @640 | peft (`yoloe_peft_640`) | 0.02% (11.6K / 58.8M) | 0.310 | 0.314 | -0.401 [-0.450, -0.338] | below | — |
| YOLOE-26x @640 | full (`yoloe_full_640`) | 100.0% (58.8M / 58.8M) | 0.708 | 0.737 | +0.022 [+0.008, +0.049] | above | — |
| YOLO-World-v2-x @1280 | peft (`yoloworld_peft`) | 12.6% (9.2M / 72.9M) | 0.685 | 0.711 | -0.005 [-0.042, +0.056] | tie | 0.189 → 0.711 (+0.521, test) |
| YOLO-World-v2-x @1280 | full (`yoloworld_full`) | 100.0% (72.9M / 72.9M) | 0.762 | 0.776 | +0.061 [+0.043, +0.076] | above | 0.189 → 0.776 (+0.587, test) |
| OWLv2-L/14 | peft | 1.1% (5.0M / 437.6M) | 0.557 | 0.541 | -0.175 [-0.254, -0.078] | below | 0.315 → 0.541 (+0.226, test) |
| OWLv2-L/14 | full | 71.6% (313.2M / 437.6M) | 0.624 | 0.612 | -0.103 [-0.175, -0.014] | below | 0.315 → 0.612 (+0.297, test) |
| DEIMv2-M | peft | 47.0% (8.6M / 18.2M) | 0.682 | 0.670 | -0.046 [-0.078, -0.002] | below | — |
| DEIMv2-M | full | 100.0% (18.2M / 18.2M) | 0.688 | 0.701 | -0.014 [-0.034, +0.025] | tie | — |
| DEIMv2-L | peft (`l_peft`) | 33.8% (10.8M / 32.1M) | 0.683 | 0.668 | -0.048 [-0.104, +0.004] | tie | — |
| DEIMv2-L | full (`l_full`) | 100.0% (32.1M / 32.1M) | 0.705 | 0.695 | -0.021 [-0.053, +0.024] | tie | — |
| Florence-2-large | peft | 0.9% (6.9M / 781.2M) | 0.512 | 0.499 | -0.216 [-0.284, -0.111] | below | 0.238 → 0.499 (+0.261, test) |
| Florence-2-large | full | 100.0% (776.5M / 776.5M) | 0.672 | 0.655 | -0.060 [-0.100, +0.004] | tie | 0.238 → 0.655 (+0.417, test) |
| Qwen3-VL (peft 8B, full 4B) | peft | 0.2% (15.3M / 8.78B) | 0.517 | 0.548 | -0.168 [-0.201, -0.094] | below | 0.318 → 0.548 (+0.230, test) |
| Qwen3-VL (peft 8B, full 4B) | full | 90.6% (4.02B / 4.44B) | 0.539 | 0.541 | -0.175 [-0.231, -0.093] | below | 0.318 → 0.541 (+0.223, test) |
| PaliGemma 2 3B-448 | peft | 0.3% (9.0M / 3.04B) | 0.293 | 0.278 | -0.437 [-0.486, -0.395] | below | 0.008 → 0.293 (+0.285, val) |
| PaliGemma 2 3B-448 | full | 86.4% (2.62B / 3.03B) | 0.268 | 0.265 | -0.451 [-0.500, -0.404] | below | 0.008 → 0.268 (+0.261, val) |
| SmolVLM2-2.2B | peft | 1.2% (27.5M / 2.27B) | 0.001 | 0.000 | -0.715 [-0.737, -0.702] | below | 0.000 → 0.001 (+0.001, val) |
| SmolVLM2-2.2B | full | 81.6% (1.83B / 2.25B) | 0.001 | 0.000 | -0.715 [-0.737, -0.702] | below | 0.000 → 0.001 (+0.001, val) |
<!-- TABLE:ft_main END -->

- **Trainable** is the share of parameters the arm updates (from each run's
  `meta.json`). `peft` is the light arm; what it trains differs by family (see
  [PEFT vs full](#peft-vs-full-fine-tune-by-family)).
- **Val** chose the checkpoint (and, for some models, the learning rate). **Test** was
  predicted once.
- **Δ vs YOLO26m** is the paired difference on the same 94 test images, with the
  clip-clustered 95% interval.
- **Zero-shot → fine-tuned** compares with the same checkpoint before fine-tuning. A
  `test` entry is the published zero-shot row from VLM_VS_FINETUNED.md; a `val` entry
  is a val-only zero-shot run in the fine-tuning repo (MM-Grounding-DINO, SAM 3 and
  PaliGemma 2 have no published row; SmolVLM2's was removed). Several published
  zero-shot rows ran a different pipeline from the fine-tuned arm (2x2 tiles,
  constant confidence, upscaling; see the per-model notes), so the gain mixes
  fine-tuning with the pipeline change.

### Per clip

<!-- TABLE:ft_per_clip START -->
| Row | boston-celtics-new-york-knicks-game-1-q1|07_41-07_34 | boston-celtics-new-york-knicks-game-4-q1|05_06-05_01 | boston-celtics-orlando-magic-game-4-q1|11_44-11_36 |
| --- | --- | --- | --- |
| YOLO26m | 0.713 | 0.737 | 0.702 |
| DEIM-M | 0.738 | 0.709 | 0.626 |
| LLMDet-large, peft | 0.817 | 0.782 | 0.739 |
| LLMDet-large, full | 0.814 | 0.783 | 0.766 |
| MM-Grounding-DINO-B, peft | 0.805 | 0.749 | 0.709 |
| MM-Grounding-DINO-B, full | 0.798 | 0.745 | 0.704 |
| SAM 3, peft | 0.764 | 0.717 | 0.637 |
| SAM 3, full | 0.777 | 0.762 | 0.682 |
| YOLOE-26x @1280, peft (`yoloe_peft`) | 0.488 | 0.391 | 0.386 |
| YOLOE-26x @1280, full (`yoloe_full`) | 0.802 | 0.798 | 0.738 |
| YOLOE-26x @640, peft (`yoloe_peft_640`) | 0.378 | 0.286 | 0.295 |
| YOLOE-26x @640, full (`yoloe_full_640`) | 0.764 | 0.752 | 0.711 |
| YOLO-World-v2-x @1280, peft (`yoloworld_peft`) | 0.772 | 0.713 | 0.660 |
| YOLO-World-v2-x @1280, full (`yoloworld_full`) | 0.790 | 0.804 | 0.745 |
| OWLv2-L/14, peft | 0.638 | 0.580 | 0.446 |
| OWLv2-L/14, full | 0.702 | 0.642 | 0.525 |
| DEIMv2-M, peft | 0.713 | 0.684 | 0.624 |
| DEIMv2-M, full | 0.740 | 0.703 | 0.674 |
| DEIMv2-L, peft (`l_peft`) | 0.719 | 0.696 | 0.596 |
| DEIMv2-L, full (`l_full`) | 0.739 | 0.718 | 0.648 |
| Florence-2-large, peft | 0.607 | 0.521 | 0.416 |
| Florence-2-large, full | 0.720 | 0.675 | 0.601 |
| Qwen3-VL (peft 8B, full 4B), peft | 0.622 | 0.537 | 0.502 |
| Qwen3-VL (peft 8B, full 4B), full | 0.624 | 0.559 | 0.470 |
| PaliGemma 2 3B-448, peft | 0.320 | 0.314 | 0.214 |
| PaliGemma 2 3B-448, full | 0.310 | 0.310 | 0.201 |
| SmolVLM2-2.2B, peft | 0.000 | 0.000 | 0.000 |
| SmolVLM2-2.2B, full | 0.000 | 0.000 | 0.001 |
<!-- TABLE:ft_per_clip END -->

With three clips, a row is "above" YOLO26m only if it wins on every clip by enough
that no resample of clips reverses it.

### Against DEIM-M, and the image-level contrast

<!-- TABLE:ft_contrast START -->
| Row | Test | Δ vs YOLO26m, clustered CI | image-level CI | clips ahead | Δ vs DEIM-M, clustered CI | image-level CI | clips ahead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLMDet-large, peft | 0.776 | +0.060 [+0.036, +0.102] | [+0.044, +0.077] | 3 of 3 | +0.090 [+0.072, +0.112] | [+0.075, +0.105] | 3 of 3 |
| LLMDet-large, peft, seed 1 | 0.770 | +0.054 [+0.023, +0.103] | [+0.041, +0.068] | 3 of 3 | +0.084 [+0.066, +0.098] | [+0.070, +0.097] | 3 of 3 |
| LLMDet-large, peft, seed 2 | 0.786 | +0.070 [+0.049, +0.105] | [+0.058, +0.083] | 3 of 3 | +0.099 [+0.077, +0.126] | [+0.086, +0.113] | 3 of 3 |
| LLMDet-large, full | 0.785 | +0.070 [+0.046, +0.100] | [+0.057, +0.084] | 3 of 3 | +0.099 [+0.074, +0.137] | [+0.084, +0.114] | 3 of 3 |
| MM-Grounding-DINO-B, peft | 0.749 | +0.034 [+0.007, +0.089] | [+0.022, +0.048] | 3 of 3 | +0.063 [+0.040, +0.082] | [+0.051, +0.076] | 3 of 3 |
| MM-Grounding-DINO-B, peft, seed 1 | 0.761 | +0.046 [+0.015, +0.091] | [+0.033, +0.058] | 3 of 3 | +0.075 [+0.043, +0.104] | [+0.062, +0.089] | 3 of 3 |
| MM-Grounding-DINO-B, peft, seed 2 | 0.747 | +0.031 [-0.009, +0.081] | [+0.015, +0.047] | 2 of 3 | +0.060 [+0.046, +0.066] | [+0.046, +0.075] | 3 of 3 |
| MM-Grounding-DINO-B, full | 0.744 | +0.029 [+0.002, +0.082] | [+0.015, +0.043] | 3 of 3 | +0.058 [+0.036, +0.077] | [+0.045, +0.073] | 3 of 3 |
| SAM 3, peft | 0.702 | -0.014 [-0.064, +0.048] | [-0.029, +0.003] | 1 of 3 | +0.016 [+0.008, +0.026] | [+0.003, +0.029] | 3 of 3 |
| SAM 3, full | 0.737 | +0.021 [-0.019, +0.062] | [+0.007, +0.036] | 2 of 3 | +0.050 [+0.039, +0.058] | [+0.036, +0.065] | 3 of 3 |
| YOLOE-26x @1280, peft (`yoloe_peft`) | 0.414 | -0.301 [-0.345, -0.228] | [-0.318, -0.283] | 0 of 3 | -0.272 [-0.316, -0.241] | [-0.289, -0.256] | 0 of 3 |
| YOLOE-26x @1280, full (`yoloe_full`) | 0.775 | +0.060 [+0.037, +0.088] | [+0.046, +0.073] | 3 of 3 | +0.089 [+0.066, +0.112] | [+0.074, +0.105] | 3 of 3 |
| YOLOE-26x @640, peft (`yoloe_peft_640`) | 0.314 | -0.401 [-0.450, -0.338] | [-0.418, -0.383] | 0 of 3 | -0.372 [-0.421, -0.332] | [-0.389, -0.355] | 0 of 3 |
| YOLOE-26x @640, full (`yoloe_full_640`) | 0.737 | +0.022 [+0.008, +0.049] | [+0.009, +0.034] | 3 of 3 | +0.051 [+0.027, +0.084] | [+0.035, +0.067] | 3 of 3 |
| YOLO-World-v2-x @1280, peft (`yoloworld_peft`) | 0.711 | -0.005 [-0.042, +0.056] | [-0.019, +0.012] | 1 of 3 | +0.024 [+0.005, +0.034] | [+0.012, +0.037] | 3 of 3 |
| YOLO-World-v2-x @1280, full (`yoloworld_full`) | 0.776 | +0.061 [+0.043, +0.076] | [+0.050, +0.073] | 3 of 3 | +0.090 [+0.054, +0.118] | [+0.076, +0.105] | 3 of 3 |
| OWLv2-L/14, peft | 0.541 | -0.175 [-0.254, -0.078] | [-0.194, -0.151] | 0 of 3 | -0.145 [-0.179, -0.101] | [-0.164, -0.126] | 0 of 3 |
| OWLv2-L/14, peft, seed 1 | 0.539 | -0.176 [-0.254, -0.068] | [-0.195, -0.152] | 0 of 3 | -0.147 [-0.178, -0.091] | [-0.165, -0.126] | 0 of 3 |
| OWLv2-L/14, peft, seed 2 | 0.541 | -0.174 [-0.254, -0.077] | [-0.194, -0.150] | 0 of 3 | -0.145 [-0.179, -0.099] | [-0.165, -0.123] | 0 of 3 |
| OWLv2-L/14, full | 0.612 | -0.103 [-0.175, -0.014] | [-0.122, -0.081] | 0 of 3 | -0.074 [-0.100, -0.037] | [-0.089, -0.057] | 0 of 3 |
| OWLv2-L/14, full, seed 1 | 0.638 | -0.077 [-0.152, +0.013] | [-0.095, -0.058] | 1 of 3 | -0.048 [-0.077, -0.009] | [-0.063, -0.032] | 0 of 3 |
| DEIMv2-M, peft | 0.670 | -0.046 [-0.078, -0.002] | [-0.060, -0.030] | 1 of 3 | -0.016 [-0.025, -0.003] | [-0.027, -0.005] | 0 of 3 |
| DEIMv2-M, full | 0.701 | -0.014 [-0.034, +0.025] | [-0.026, -0.001] | 1 of 3 | +0.015 [-0.006, +0.046] | [+0.004, +0.026] | 2 of 3 |
| DEIMv2-L, peft (`l_peft`) | 0.668 | -0.048 [-0.104, +0.004] | [-0.062, -0.033] | 1 of 3 | -0.018 [-0.030, -0.014] | [-0.029, -0.008] | 0 of 3 |
| DEIMv2-L, full (`l_full`) | 0.695 | -0.021 [-0.053, +0.024] | [-0.035, -0.008] | 1 of 3 | +0.008 [0.000, +0.022] | [-0.004, +0.020] | 3 of 3 |
| Florence-2-large, peft | 0.499 | -0.216 [-0.284, -0.111] | [-0.238, -0.191] | 0 of 3 | -0.187 [-0.209, -0.133] | [-0.204, -0.165] | 0 of 3 |
| Florence-2-large, full | 0.655 | -0.060 [-0.100, +0.004] | [-0.076, -0.041] | 1 of 3 | -0.031 [-0.035, -0.019] | [-0.048, -0.011] | 0 of 3 |
| Florence-2-large, `lora_all64`: LoRA r64, all language-side linear layers | 0.492 | -0.224 [-0.283, -0.124] | [-0.246, -0.200] | 0 of 3 | -0.195 [-0.217, -0.146] | [-0.212, -0.172] | 0 of 3 |
| Qwen3-VL (peft 8B, full 4B), peft | 0.548 | -0.168 [-0.201, -0.094] | [-0.189, -0.143] | 0 of 3 | -0.139 [-0.171, -0.116] | [-0.157, -0.118] | 0 of 3 |
| Qwen3-VL (peft 8B, full 4B), full | 0.541 | -0.175 [-0.231, -0.093] | [-0.195, -0.152] | 0 of 3 | -0.146 [-0.156, -0.115] | [-0.164, -0.127] | 0 of 3 |
| Qwen3-VL (peft 8B, full 4B), `peft_4b`: LoRA on the 4B (like-for-like with full) | 0.533 | -0.182 [-0.230, -0.093] | [-0.203, -0.159] | 0 of 3 | -0.153 [-0.180, -0.115] | [-0.172, -0.135] | 0 of 3 |
| PaliGemma 2 3B-448, peft | 0.278 | -0.437 [-0.486, -0.395] | [-0.454, -0.417] | 0 of 3 | -0.408 [-0.419, -0.396] | [-0.426, -0.388] | 0 of 3 |
| PaliGemma 2 3B-448, full | 0.265 | -0.451 [-0.500, -0.404] | [-0.468, -0.430] | 0 of 3 | -0.422 [-0.435, -0.400] | [-0.437, -0.403] | 0 of 3 |
| SmolVLM2-2.2B, peft | 0.000 | -0.715 [-0.737, -0.702] | [-0.728, -0.704] | 0 of 3 | -0.686 [-0.736, -0.628] | [-0.703, -0.671] | 0 of 3 |
| SmolVLM2-2.2B, full | 0.000 | -0.715 [-0.737, -0.702] | [-0.728, -0.704] | 0 of 3 | -0.686 [-0.736, -0.628] | [-0.703, -0.671] | 0 of 3 |
<!-- TABLE:ft_contrast END -->

## Seeds

<!-- TABLE:ft_seeds START -->
| Model | Arm | Seeds | Val by seed | Test by seed | Test mean ± sd | Test range | Other arm, seed 0 (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLMDet-large | peft | 0 / 1 / 2 | 0.739 / 0.731 / 0.734 | 0.776 / 0.770 / 0.786 | 0.777 ± 0.008 | 0.016 | full 0.785 |
| MM-Grounding-DINO-B | peft | 0 / 1 / 2 | 0.713 / 0.719 / 0.717 | 0.749 / 0.761 / 0.747 | 0.752 ± 0.008 | 0.014 | full 0.744 |
| OWLv2-L/14 | peft | 0 / 1 / 2 | 0.557 / 0.556 / 0.556 | 0.541 / 0.539 / 0.541 | 0.541 ± 0.001 | 0.002 | full 0.612 |
| OWLv2-L/14 | full | 0 / 1 | 0.624 / 0.647 | 0.612 / 0.638 | 0.625 ± 0.018 | 0.026 | peft 0.541 |
<!-- TABLE:ft_seeds END -->

The sd is across training seeds on a fixed test set. It does not include the test-set
sampling noise in the clustered intervals above, which is the larger term. Every other
arm in this report is a single seed (0).

## PEFT vs full fine-tune, by family

<!-- TABLE:ft_peft_vs_full START -->
| Family | Model | Light arm trains | Light test | Middle level (test) | Full test | Full - light, val | Full - light, test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen backbone + DETR head | DEIMv2-M | 47.0% (8.6M / 18.2M) | 0.670 | — | 0.701 | +0.006 | +0.031 |
| frozen backbone + DETR head | DEIMv2-L | 33.8% (10.8M / 32.1M) | 0.668 | — | 0.695 | +0.022 | +0.027 |
| generative VLM | Florence-2-large | 0.9% (6.9M / 781.2M) | 0.499 | `lora_all64` 4.5% (36.8M / 811.1M): 0.492 | 0.655 | +0.159 | +0.156 |
| generative VLM | Qwen3-VL (peft 8B, full 4B) | 0.2% (15.3M / 8.78B) | 0.548 | `peft_4b` 0.3% (11.8M / 4.45B): 0.533 | 0.541 | +0.022 | -0.007 |
| generative VLM | PaliGemma 2 3B-448 | 0.3% (9.0M / 3.04B) | 0.278 | — | 0.265 | -0.025 | -0.014 |
| generative VLM | SmolVLM2-2.2B | 1.2% (27.5M / 2.27B) | 0.000 | — | 0.000 | 0.000 | 0.000 |
| open-vocab detector | LLMDet-large | 11.4% (39.0M / 343.1M) | 0.776 | — | 0.785 | 0.000 | +0.009 |
| open-vocab detector | MM-Grounding-DINO-B | 16.0% (37.2M / 233.0M) | 0.749 | — | 0.744 | -0.001 | -0.005 |
| open-vocab detector | SAM 3 | 4.6% (38.5M / 838.2M) | 0.702 | — | 0.737 | +0.040 | +0.035 |
| open-vocab detector | YOLOE-26x @1280 | 0.02% (11.6K / 58.8M) | 0.414 | — | 0.775 | +0.374 | +0.361 |
| open-vocab detector | YOLOE-26x @640 | 0.02% (11.6K / 58.8M) | 0.314 | — | 0.737 | +0.398 | +0.423 |
| open-vocab detector | YOLO-World-v2-x @1280 | 12.6% (9.2M / 72.9M) | 0.711 | — | 0.776 | +0.077 | +0.066 |
| open-vocab detector | OWLv2-L/14 | 1.1% (5.0M / 437.6M) | 0.541 | — | 0.612 | +0.066 | +0.071 |
<!-- TABLE:ft_peft_vs_full END -->

- **Open-vocab detectors.** When the light arm freezes only the image backbone and
  text encoder and trains the fusion encoder, decoder and heads (LLMDet,
  MM-Grounding-DINO), it matches the full fine-tune. SAM 3's light arm trains the
  fusion encoder and decoder too, from a smaller share of its parameters, and lands a
  little under its full arm. When the light arm is only the output heads, it falls
  well short: OWLv2 trains its detection heads, and the official YOLOE linear probe
  trains one 1x1 conv per class branch and cannot move a box. YOLO-World has no
  official probe; its light arm trains the whole detection head and sits between.
  So "light ≈ full" holds for the grounding detectors whose light arm includes the
  fusion and decoder, not for head-only probes.
- **Frozen backbone + DETR head (DEIMv2).** Freezing the DINOv3 backbone costs a few
  points at both sizes, mostly on `ball`.
- **Generative VLMs.** Florence-2's LoRA arm is far below its full fine-tune. The
  `lora_all64` middle level (5x the LoRA parameters, MLPs included) lands where the
  attention-only LoRA does, so language-side LoRA capacity is not the bottleneck. The
  frozen DaViT vision tower is, above all for the ball. Qwen3-VL's LoRA matches full
  fine-tuning of the language model; the 4B LoRA control shows this holds at matched
  size, not only because the LoRA arm is the larger 8B. PaliGemma 2's two arms are both
  low and close together; its 448 px squash, not the arm, limits it. SmolVLM2 learned
  the output format and nothing else in either arm.

## The YOLOE resolution confound

<!-- TABLE:ft_resolution START -->
| Model | Arm | Input | Val | Test | ball AP50 | number AP50 | Δ vs YOLO26m [clustered CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOE-26x @1280 | yoloe_peft | 1280 | 0.400 | 0.414 | 0.668 | 0.397 | -0.301 [-0.345, -0.228] |
| YOLOE-26x @1280 | yoloe_full | 1280 | 0.774 | 0.775 | 0.879 | 0.972 | +0.060 [+0.037, +0.088] |
| YOLOE-26x @640 | yoloe_peft_640 | 640 | 0.310 | 0.314 | 0.386 | 0.012 | -0.401 [-0.450, -0.338] |
| YOLOE-26x @640 | yoloe_full_640 | 640 | 0.708 | 0.737 | 0.862 | 0.949 | +0.022 [+0.008, +0.049] |
| YOLO-World-v2-x @1280 | yoloworld_peft | 1280 | 0.685 | 0.711 | 0.734 | 0.940 | -0.005 [-0.042, +0.056] |
| YOLO-World-v2-x @1280 | yoloworld_full | 1280 | 0.762 | 0.776 | 0.886 | 0.984 | +0.061 [+0.043, +0.076] |
<!-- TABLE:ft_resolution END -->

YOLOE-26x and YOLO-World-v2-x ran their main arms at 1280 to match their zero-shot rows.
Every detector in FINAL_COMPARISON_640.md ran at 640. The 640 YOLOE arms were added to
separate the two effects: compare **YOLOE @640** with YOLO26m, not @1280. Most of what
1280 adds is `ball` and `number`, the small classes. YOLO-World was not re-run at 640.

The same caveat applies, more weakly, to the other foundation rows: LLMDet and
MM-Grounding-DINO run at short side 800, SAM 3 and OWLv2 at 1008, Qwen3-VL at native
1080p. All of them see more pixels than the 640 detectors, and none of the published
detectors was retrained at a higher resolution. The input of every row is in the
latency table below.

## Per-class AP@50 (test)

<!-- TABLE:ft_per_class START -->
| Row | mAP@50 | player | ball | referee | rim | number |
| --- | --- | --- | --- | --- | --- | --- |
| LLMDet-large, peft | 0.971 | 0.976 | 0.927 | 0.988 | 0.998 | 0.968 |
| LLMDet-large, peft, seed 1 | 0.971 | 0.978 | 0.925 | 0.994 | 1.000 | 0.957 |
| LLMDet-large, peft, seed 2 | 0.970 | 0.984 | 0.913 | 0.992 | 0.999 | 0.960 |
| LLMDet-large, full | 0.971 | 0.979 | 0.919 | 0.990 | 1.000 | 0.965 |
| MM-Grounding-DINO-B, peft | 0.970 | 0.992 | 0.917 | 0.978 | 1.000 | 0.965 |
| MM-Grounding-DINO-B, peft, seed 1 | 0.974 | 0.988 | 0.930 | 0.981 | 1.000 | 0.970 |
| MM-Grounding-DINO-B, peft, seed 2 | 0.972 | 0.989 | 0.917 | 0.984 | 1.000 | 0.968 |
| MM-Grounding-DINO-B, full | 0.964 | 0.982 | 0.913 | 0.967 | 1.000 | 0.959 |
| SAM 3, peft | 0.928 | 0.989 | 0.848 | 0.992 | 0.959 | 0.852 |
| SAM 3, full | 0.963 | 0.990 | 0.924 | 0.985 | 1.000 | 0.916 |
| YOLOE-26x @1280, peft (`yoloe_peft`) | 0.639 | 0.971 | 0.668 | 0.843 | 0.313 | 0.397 |
| YOLOE-26x @1280, full (`yoloe_full`) | 0.964 | 0.990 | 0.879 | 0.979 | 1.000 | 0.972 |
| YOLOE-26x @640, peft (`yoloe_peft_640`) | 0.446 | 0.959 | 0.386 | 0.846 | 0.024 | 0.012 |
| YOLOE-26x @640, full (`yoloe_full_640`) | 0.955 | 0.988 | 0.862 | 0.977 | 1.000 | 0.949 |
| YOLO-World-v2-x @1280, peft (`yoloworld_peft`) | 0.923 | 0.988 | 0.734 | 0.956 | 0.998 | 0.940 |
| YOLO-World-v2-x @1280, full (`yoloworld_full`) | 0.966 | 0.990 | 0.886 | 0.972 | 1.000 | 0.984 |
| OWLv2-L/14, peft | 0.884 | 0.971 | 0.799 | 0.953 | 0.823 | 0.874 |
| OWLv2-L/14, peft, seed 1 | 0.876 | 0.971 | 0.790 | 0.957 | 0.799 | 0.861 |
| OWLv2-L/14, peft, seed 2 | 0.881 | 0.971 | 0.792 | 0.957 | 0.814 | 0.873 |
| OWLv2-L/14, full | 0.930 | 0.976 | 0.886 | 0.984 | 0.883 | 0.919 |
| OWLv2-L/14, full, seed 1 | 0.945 | 0.986 | 0.877 | 0.990 | 0.945 | 0.925 |
| DEIMv2-M, peft | 0.944 | 0.988 | 0.822 | 0.998 | 1.000 | 0.914 |
| DEIMv2-M, full | 0.961 | 0.988 | 0.889 | 0.995 | 1.000 | 0.932 |
| DEIMv2-L, peft (`l_peft`) | 0.931 | 0.988 | 0.755 | 0.990 | 1.000 | 0.923 |
| DEIMv2-L, full (`l_full`) | 0.955 | 0.988 | 0.871 | 0.986 | 1.000 | 0.930 |
| Florence-2-large, peft | 0.725 | 0.946 | 0.287 | 0.853 | 0.856 | 0.685 |
| Florence-2-large, full | 0.864 | 0.947 | 0.650 | 0.892 | 0.980 | 0.851 |
| Florence-2-large, `lora_all64`: LoRA r64, all language-side linear layers | 0.746 | 0.949 | 0.343 | 0.851 | 0.941 | 0.644 |
| Qwen3-VL (peft 8B, full 4B), peft | 0.821 | 0.948 | 0.596 | 0.863 | 0.950 | 0.747 |
| Qwen3-VL (peft 8B, full 4B), full | 0.826 | 0.943 | 0.659 | 0.857 | 0.913 | 0.758 |
| Qwen3-VL (peft 8B, full 4B), `peft_4b`: LoRA on the 4B (like-for-like with full) | 0.806 | 0.958 | 0.608 | 0.846 | 0.866 | 0.755 |
| PaliGemma 2 3B-448, peft | 0.564 | 0.864 | 0.209 | 0.731 | 0.674 | 0.343 |
| PaliGemma 2 3B-448, full | 0.544 | 0.870 | 0.214 | 0.793 | 0.602 | 0.243 |
| SmolVLM2-2.2B, peft | 0.001 | 0.006 | 0.000 | 0.001 | 0.000 | 0.000 |
| SmolVLM2-2.2B, full | 0.001 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 |
<!-- TABLE:ft_per_class END -->

## T4 latency

Two different measurements, kept in separate tables. They are not comparable.

- **TensorRT fp16, GPU compute** with device-resident IO (`trtexec --noDataTransfers`
  equivalent). This is how FINAL_COMPARISON_640.md measures the detectors. Only DEIMv2
  was exported to TensorRT.
- **Native PyTorch fp16, end to end**: batch 1, PIL image in, merged5 boxes out,
  preprocessing, decoding and generation included, two same-config runs
  (`common/latency.py` in the fine-tuning repo). Every other foundation model is timed
  this way. For the generative VLMs nearly all of the time is autoregressive decoding.

### Anchor: does the rented T4 reproduce the published numbers?

The published table was measured on a dedicated GCP T4 that no longer exists. The
fine-tuned rows were timed on rented vast.ai T4s with pip-installed TensorRT 10.3. To
check that the two set-ups agree, the published YOLOX-M and DEIM-M ONNX files were
re-timed on a vast.ai T4 with the same protocol:

<!-- TABLE:ft_latency_anchor START -->
| Model | Published (ms) | vast.ai T4 re-time (ms) | Runs (ms) | Difference |
| --- | --- | --- | --- | --- |
| YOLOX-M | 5.68 | 5.84 | 5.83 / 5.84 | +2.8% |
| DEIM-M | 6.61 | 6.67 | 6.65 / 6.67 | +0.9% |
<!-- TABLE:ft_latency_anchor END -->

Both land within a few percent of the published medians, so the DEIMv2 TensorRT rows
below sit on the same footing as the published detectors.

### TensorRT fp16 (comparable to FINAL_COMPARISON_640.md)

<!-- TABLE:ft_latency_trt START -->
| Model | Measured on | Median (ms) | Test mAP@50:95 |
| --- | --- | --- | --- |
| YOLO26m | published: dedicated GCP T4, TensorRT 10.3 | 5.85 | — |
| DEIM-M | published: dedicated GCP T4, TensorRT 10.3 | 6.61 | — |
| DEIM-M | re-timed: vast.ai T4, pip TensorRT 10.3 (anchor) | 6.67 | — |
| YOLOX-M | published: dedicated GCP T4, TensorRT 10.3 | 5.68 | — |
| YOLOX-M | re-timed: vast.ai T4, pip TensorRT 10.3 (anchor) | 5.84 | — |
| RF-DETR-M | published: dedicated GCP T4, TensorRT 10.3 | 7.71 | — |
| RTMDet-M | published: dedicated GCP T4, TensorRT 10.3 | 8.19 | — |
| DAMO-YOLO-M | published: dedicated GCP T4, TensorRT 10.3 | 6.70 | — |
| RT-DETRv2-M | published: dedicated GCP T4, TensorRT 10.3 | 7.93 | — |
| DEIMv2-M (`full`) | vast.ai T4, pip TensorRT 10.3 | 11.08 | 0.701 |
| DEIMv2-L (`l_full`) | vast.ai T4, pip TensorRT 10.3 | 13.16 | 0.695 |
<!-- TABLE:ft_latency_trt END -->

### Native PyTorch fp16, end to end (not comparable to the table above)

<!-- TABLE:ft_latency_pytorch START -->
| Model | Arm timed | Input | Median (ms) | Tail (ms) | Test mAP@50:95 |
| --- | --- | --- | --- | --- | --- |
| LLMDet-large | peft | short side 800, long cap 1333 | 734.9 | p90 743.7 | 0.776 |
| LLMDet-large | full | short side 800, long cap 1333 | 743.5 | p90 752.2 | 0.785 |
| MM-Grounding-DINO-B | peft | short side 800 (1422x800) | 466.2 | p90 484.9 | 0.749 |
| SAM 3 | full | 1008x1008 | 1137.5 | p90 1141.8 | 0.737 |
| YOLOE-26x @1280 | yoloe_full | 1280 | 57.3 | p90 58.8 | 0.775 |
| YOLOE-26x @640 | yoloe_full_640 | 640 | 28.9 | p90 36.4 | 0.737 |
| YOLO-World-v2-x @1280 | yoloworld_full | 1280 | 80.6 | p90 81.7 | 0.776 |
| OWLv2-L/14 | peft | 1008x1008 (padded square) | 1825.9 | p90 1929.2 | 0.541 |
| DEIMv2-M | full | 640 | 66.0 | p90 72.9 | 0.701 |
| DEIMv2-L | l_full | 640 | 70.6 | p90 79.4 | 0.695 |
| Florence-2-large | full | 768x768 (squashed) | 1298.1 | p90 1452.7 | 0.655 |
| Florence-2-large | peft | 768x768 (squashed) | 1274.6 | p90 1369.5 | 0.499 |
| Qwen3-VL (peft 8B, full 4B) | unspecified | native 1920x1088 | 74712.4 | p90 78616.2 | — |
| PaliGemma 2 3B-448 | peft | 448x448 (squashed) | 9924.7 | p90 11432.0 | 0.278 |
| SmolVLM2-2.2B | peft | SigLIP tiles, pixel-shuffled | 21422.8 | p90 25776.1 | 0.000 |
<!-- TABLE:ft_latency_pytorch END -->

A PyTorch row with the same architecture as another arm (LoRA merged, or a frozen
backbone) has the same graph, so only one arm per model was timed unless two are
listed. The PyTorch path includes host-side work the TensorRT numbers exclude (DEIMv2
is in both tables to show the size of that gap). OWLv2's time includes the HF image
processor, which runs on the CPU in numpy (`latency_breakdown_t4.json` in the
fine-tuning repo splits it out).

## Cost

<!-- TABLE:ft_cost START -->
| Slug | Models | vast.ai spend ($) | Train GPU-h (scored runs) | Source |
| --- | --- | --- | --- | --- |
| yoloe | YOLO-World-v2-x @1280, YOLOE-26x @1280, YOLOE-26x @640 | 1.11 | 1.2 | docs/plans/yoloe.md, Cost table (all six arms, one failed box, T4) |
| llmdet | LLMDet-large | 3.80 | 5.1 | docs/plans/llmdet.md: Cost table about $2.9 (both arms, seed 1, T4) + seed 2 section about $0.9 |
| mmgdino | MM-Grounding-DINO-B | 3.97 | 6.2 | docs/plans/mmgdino.md: main arms + T4 about $2.31, peft seeds 1-2 about $1.66 |
| owlv2 | OWLv2-L/14 | 4.64 | 3.8 | docs/plans/owlv2.md: A6000 7.74 h (both arms, sweeps, seeds) $4.47 + three T4 rentals $0.17 |
| sam3 | SAM 3 | 2.75 | 2.9 | docs/plans/sam3.md, Cost table (A100 both arms + T4) |
| deimv2 | DEIMv2-L, DEIMv2-M | 2.49 | 4.8 | docs/plans/deimv2.md, Cost table (M box, L box, T4) |
| florence2 | Florence-2-large | 1.49 | 1.9 | docs/plans/florence2.md: Cost table ~1.04 (both arms, 1024 exploration, T4) + lora_all64 box 0.45 |
| qwen3vl | Qwen3-VL (peft 8B, full 4B) | 6.50 | 3.7 | docs/plans/qwen3vl.md, Cost table: A100 5.86 + failed T4 0.13 + T4 latency 0.51 |
| paligemma2 | PaliGemma 2 3B-448 | 2.75 | 1.8 | docs/plans/paligemma2.md, Cost table (A100 + T4) |
| smolvlm2 | SmolVLM2-2.2B | 2.42 | 6.0 | docs/plans/smolvlm2.md: A6000 4.99 h $2.11 + T4 1.39 h $0.31 |
| **total** |  | **31.92** |  |  |
<!-- TABLE:ft_cost END -->

Spend is the whole vast.ai bill per model: training boxes, failed rentals, exploration
runs and T4 latency boxes. It is taken from each model's plan doc, or from `meta.json`
where the plan doc has no total, and the source column says which. Per-arm training
hours and $ are in the results file, but arms that shared one GPU (LLMDet, SAM 3,
SmolVLM2, DEIMv2) report overlapping wall-clock, so per-arm values must not be summed;
use the spend column. The fine-tuning repo's running account log also includes a box
that was not part of this project.

## Deviations and caveats, per model

Taken from each model's plan doc in the fine-tuning repo (`docs/plans/<slug>.md`) and
its `meta.json`.

- **LLMDet-large.** Trained through the HF transformers port
  (`MMGroundingDinoForObjectDetection`), not upstream mmdet; same weights. Test runs a
  single pass at short side 800 / long side 1333 with no tiling, while the zero-shot
  row used 2x2 tiles. Native 1080p did not fit at batch 4 on 80 GB. Early stopping
  (5-epoch patience, 20-epoch cap) was a rule set by the lead during the run. Both arms
  trained concurrently on one A100.
- **MM-Grounding-DINO-B.** HF port, custom training loop. The full arm was run twice
  (lr 1e-4 and 5e-5); the lr 1e-4 run diverged late but won on val by a noise-level
  margin and is the only full checkpoint predicted on test. The zero-shot reference is
  a val-only run of this checkpoint; the published Grounding-DINO row is a different
  checkpoint (IDEA `grounding-dino-base`). No plan doc in the fine-tuning repo; facts
  are from `meta.json`.
- **SAM 3.** Upstream's trainer cannot train the image trunk at the pinned commit (its
  fused ViT MLP refuses autograd), so `patch_vit_mlp` replaces it in every train,
  predict and latency run. Upstream's RF100-VL fine-tune recipe, with every change
  marked. Arms trained concurrently. No published zero-shot row; the reference is
  val-only. Latency timed on the full arm only (same graph), fp16 autocast because the
  T4 has no bf16.
- **YOLOE-26x / YOLO-World-v2-x.** 1280 for the main arms (see the resolution
  section). Class names are the zero-shot prompts. The light-arm learning rate was
  chosen on val from the docs default and one higher value. The YOLOE light arm is
  Ultralytics' official linear probe; YOLO-World has none, so its light arm trains the
  detection head. `yoloe_full` ran at a per-step batch of 4 after an OOM (nominal batch
  unchanged). Checkpoints were selected by Ultralytics' val fitness, not this scorer.
  The zero-shot rows used the VLM post-processing filters; the fine-tuned rows do not.
  AGPL-3.0.
- **OWLv2-L/14.** 1008 px (the large/14 native size). HF ships no OWLv2 training loss,
  so the fine-tuning repo implements the OWL-ViT set loss. The light-arm head learning
  rate was picked from a five-config sweep on val; the full arm is one configuration
  whose best epoch was its last. No tiling, while the zero-shot row tiles. The extra
  light-arm seeds have no `meta.json` yet; their params are taken from the seed-0 arm.
- **DEIMv2-M / -L.** The light arm freezes the DINOv3 backbone (no LoRA), per the PRD.
  Schedule compressed as for the published DEIM-M row. ViT interaction indexes were
  picked on val. PyTorch latency uses `torch.autocast` because a fully halved model
  breaks DINOv3's RoPE. The M ONNX was traced at batch 1 (batch 32 was OOM-killed);
  the L export freezes the RoPE to its 640 px value so TensorRT 10.3 can parse it. The
  M engine was timed while the L engine compiled on the CPU (GPU idle). No zero-shot
  row: DEIMv2 is a closed-set COCO detector.
- **Florence-2-large.** The light arm also trains the vision-to-language connector.
  Fine-tuned arms use the `<OD>` task on the whole 768x768 (squashed) frame with
  token-probability confidences; the zero-shot row used phrase grounding, 2x2 tiles
  and a constant confidence. Greedy decoding for both arms. **Disclosure:** the full
  arm's beam-3 val score was read only after its test predictions were made; test was
  not re-predicted, so the published test number is greedy. 1024 px was tried and
  dropped on val. `lora_all64` is a follow-up middle level whose best val was its last
  eval.
- **Qwen3-VL.** The full arm is the 4B, because a full fine-tune of the 8B with fp32
  master weights does not fit one 80 GB GPU; `peft_4b` was added for a like-for-like
  comparison. Native resolution, not the zero-shot row's 2x upscale; the published
  zero-shot row also used a constant confidence. A second full run (6 epochs) was
  launched after looking at val and lost on val. Val/test predictions use vLLM;
  latency uses HF `generate`.
- **PaliGemma 2 3B-448.** 448x448 squash, so the ball and jersey numbers are a few
  pixels. The light arm also trains the projector. Arms ran sequentially (concurrent
  OOM). Greedy generation gives no low-confidence tail of extra boxes, which caps mAP
  relative to detectors. The zero-shot `mix-448` row is val-only and new. Only the
  light arm was timed. Gemma licence (gated).
- **SmolVLM2-2.2B.** **Manual stop after epoch 8**: val sat at the floor for four
  consecutive evaluations with no trend, so both arms were stopped instead of running
  to the 12-epoch cap. The full arm keeps the SigLIP vision tower frozen. The light
  arm is QLoRA (4-bit base) with an unquantised trainable connector. The zero-shot
  reference is val-only and every box was zero-area. Latency (one pass, light arm)
  is pending.

## Reproducing these numbers

**From this repo.** The test and val dumps of every row are committed in
`results/finetune/` (large ones gzipped). Two scripts rebuild everything else:

```bash
pixi run python scripts/write_finetune_results.py   # rescore dumps, read meta/latency
pixi run python scripts/run_finetune_bootstrap.py   # clustered + image-level CIs
pixi run python scripts/generate_report.py --write --report foundation_finetune
```

`write_finetune_results.py` re-copies the dumps from the fine-tuning repo; to rescore
only what is committed, decompress a dump (`gunzip -k`) and run
`scripts/score_finetune_dump.py --pred <dump> --split test --data-root <dataset>`.
The bootstrap also needs the YOLO26m and DEIM-M test predictions that
`reproduction_640.yaml` points at, as for the published intervals.

**Training.** The training code is in the separate `foundation-finetune` repo, and every
fine-tuned checkpoint is in a private Hugging Face repo (`ortizeg/ffm-<slug>-<arm>`).
Neither is public yet. Whether a checkpoint can be published depends on the licence of
its base weights:

<!-- TABLE:ft_licence START -->
| Model | Licence of the base weights | Fine-tuned weights redistributable here? |
| --- | --- | --- |
| LLMDet-large | Apache-2.0 | licence allows it; not published yet (private HF repo) |
| MM-Grounding-DINO-B | Apache-2.0 | licence allows it; not published yet (private HF repo) |
| SAM 3 | SAM License (gated) | no: gated licence, private HF repo only |
| YOLOE-26x @1280 | AGPL-3.0 | no: AGPL-3.0 weights are not redistributed by this repo |
| YOLOE-26x @640 | AGPL-3.0 | no: AGPL-3.0 weights are not redistributed by this repo |
| YOLO-World-v2-x @1280 | AGPL-3.0 | no: AGPL-3.0 weights are not redistributed by this repo |
| OWLv2-L/14 | Apache-2.0 | licence allows it; not published yet (private HF repo) |
| DEIMv2-M | Apache-2.0 code, DINOv3 License weights | no: gated licence, private HF repo only |
| DEIMv2-L | Apache-2.0 code, DINOv3 License weights | no: gated licence, private HF repo only |
| Florence-2-large | MIT | licence allows it; not published yet (private HF repo) |
| Qwen3-VL (peft 8B, full 4B) | Apache-2.0 | licence allows it; not published yet (private HF repo) |
| PaliGemma 2 3B-448 | Gemma Terms of Use (gated) | no: gated licence, private HF repo only |
| SmolVLM2-2.2B | Apache-2.0 | licence allows it; not published yet (private HF repo) |
<!-- TABLE:ft_licence END -->

## Pending inputs

Runs still in flight when this report was generated. Re-running the three commands
above picks them up.

<!-- TABLE:ft_pending START -->
- owlv2: owlv2/peft_seed1/meta.json
- owlv2: owlv2/peft_seed2/meta.json
- owlv2: owlv2/full_seed1/meta.json
<!-- TABLE:ft_pending END -->
