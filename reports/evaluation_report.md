# Evaluation report

Run `2026-10-27_run02` · 23 unseen test captures (2 with a PSNR target)

Test scenes were never trained on, and scenes marked `company` are test-only.

## Ranking quality

Spearman correlation between each method's score and the quality target, with a percentile bootstrap 95% interval.

| method | Spearman (95% CI) | videos | image bundles |
|---|---|---|---|
| **capturecheck** | 0.71 (0.37 – 0.91) | n/a | 0.71 |
| `frame_count` | 0.37 (-0.03 – 0.69) | n/a | 0.37 |
| `mean_blur` | 0.20 (-0.28 – 0.61) | n/a | 0.20 |
| `orb_matches` | 0.47 (0.04 – 0.81) | n/a | 0.47 |
| `lightgbm_features` | 0.60 (0.22 – 0.85) | n/a | 0.60 |

## Problems found

Precision and recall of the model's problem heads against the labels.

| problem | test captures | positives | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `motion_blur` | 23 | 3 | 1 | 1.00 | 0.33 | 0.50 |
| `lighting` | 23 | 3 | 16 | 0.19 | 1.00 | 0.32 |
| `no_walking` | 23 | 4 | 12 | 0.17 | 0.50 | 0.25 |
| `poor_coverage` | 23 | 11 | 23 | 0.48 | 1.00 | 0.65 |
| `too_few_views` | 23 | 4 | 19 | 0.21 | 1.00 | 0.35 |
| `low_overlap` | 23 | 8 | 9 | 0.56 | 0.62 | 0.59 |
| `mixed_cameras` | 23 | 0 | 0 | n/a | n/a | n/a |
| `reflective` | 23 | 0 | 0 | n/a | n/a | n/a |
| `low_texture` | 23 | 1 | 0 | n/a | 0.00 | n/a |
| `noisy` | 23 | 3 | 3 | 1.00 | 1.00 | 1.00 |

## Runtime

- Measuring a capture: **1.08 s** (OpenCV checks + DINOv2, mean over everything labelled)
- Scoring it with the model: **0.0008 s** (onnx)
- Budget: 60 s per capture on a laptop CPU

## Predicted score vs target

![predicted vs psnr](figures/predicted_vs_psnr_2026-10-27_run02.png)

## Split

- train scenes (5): `bedroom`, `garage`, `hallway`, `office`, `studio`
- test scenes (2): `company_room`, `kitchen`

## Honest notes

- Fewer than 3 test captures have a PSNR target, so the correlation is against damage severity (clean > light > medium > heavy), not PSNR.
- A test set of this size cannot separate methods whose intervals overlap; read the CI, not the point estimate.
