# Evaluation report

Run `2026-10-27_run01` · 23 unseen test captures (2 with a PSNR target)

Test scenes were never trained on, and scenes marked `company` are test-only.

## Ranking quality

Spearman correlation between each method's score and the quality target, with a percentile bootstrap 95% interval.

| method | Spearman (95% CI) | videos | image bundles |
|---|---|---|---|
| **capturecheck** | 0.59 (0.17 – 0.86) | n/a | 0.59 |
| `frame_count` | 0.37 (-0.03 – 0.69) | n/a | 0.37 |
| `mean_blur` | 0.20 (-0.28 – 0.61) | n/a | 0.20 |
| `orb_matches` | 0.47 (0.04 – 0.81) | n/a | 0.47 |
| `lightgbm_features` | 0.55 (0.14 – 0.84) | n/a | 0.55 |

## Problems found

Precision and recall of the model's problem heads against the labels.

| problem | test captures | positives | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `motion_blur` | 23 | 3 | 3 | 1.00 | 1.00 | 1.00 |
| `lighting` | 23 | 3 | 11 | 0.27 | 1.00 | 0.43 |
| `no_walking` | 23 | 4 | 3 | 1.00 | 0.75 | 0.86 |
| `poor_coverage` | 23 | 11 | 17 | 0.59 | 0.91 | 0.71 |
| `too_few_views` | 23 | 4 | 5 | 0.60 | 0.75 | 0.67 |
| `low_overlap` | 23 | 8 | 5 | 0.80 | 0.50 | 0.62 |
| `mixed_cameras` | 23 | 0 | 0 | n/a | n/a | n/a |
| `reflective` | 23 | 0 | 0 | n/a | n/a | n/a |
| `low_texture` | 23 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| `noisy` | 23 | 3 | 3 | 1.00 | 1.00 | 1.00 |

## Runtime

- Measuring a capture: **1.22 s** (OpenCV checks + DINOv2, mean over everything labelled)
- Scoring it with the model: **0.001 s** (onnx)
- Budget: 60 s per capture on a laptop CPU

## Predicted score vs target

![predicted vs psnr](figures/predicted_vs_psnr_2026-10-27_run01.png)

## Split

- train scenes (5): `bedroom`, `garage`, `hallway`, `office`, `studio`
- test scenes (2): `company_room`, `kitchen`

## Honest notes

- Fewer than 3 test captures have a PSNR target, so the correlation is against damage severity (clean > light > medium > heavy), not PSNR.
- A test set of this size cannot separate methods whose intervals overlap; read the CI, not the point estimate.
