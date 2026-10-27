# Evaluation report

Run `2026-10-27_run01` · 17 unseen test captures (1 video, 16 image bundle; 2 with a PSNR target)

Test scenes were never trained on, and scenes marked `company` are test-only.

## Ranking quality

Spearman correlation between each method's score and the quality target, with a percentile bootstrap 95% interval.

| method | Spearman (95% CI) | videos | image bundles |
|---|---|---|---|
| **capturecheck** | 0.67 (0.31 – 0.89) | n/a | 0.72 |
| `frame_count` | 0.31 (-0.19 – 0.73) | n/a | 0.27 |
| `mean_blur` | -0.15 (-0.67 – 0.34) | n/a | -0.07 |
| `orb_matches` | 0.51 (-0.02 – 0.90) | n/a | 0.45 |
| `lightgbm_features` | 0.76 (0.45 – 0.93) | n/a | 0.71 |

## Problems found

Precision and recall of the model's problem heads against the labels.

| problem | test captures | positives | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `motion_blur` | 17 | 3 | 2 | 1.00 | 0.67 | 0.80 |
| `lighting` | 17 | 3 | 6 | 0.50 | 1.00 | 0.67 |
| `no_walking` | 17 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| `poor_coverage` | 17 | 16 | 14 | 1.00 | 0.88 | 0.93 |
| `too_few_views` | 17 | 3 | 9 | 0.33 | 1.00 | 0.50 |
| `low_overlap` | 17 | 5 | 6 | 0.83 | 1.00 | 0.91 |
| `mixed_cameras` | 17 | 0 | 0 | n/a | n/a | n/a |
| `reflective` | 17 | 0 | 0 | n/a | n/a | n/a |
| `low_texture` | 17 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| `noisy` | 17 | 3 | 4 | 0.75 | 1.00 | 0.86 |

## Runtime

- Measuring a capture: **2.17 s** (OpenCV checks + DINOv2, mean over everything labelled)
- Scoring it with the model: **0.001 s** (onnx)
- Budget: 60 s per capture on a laptop CPU

## Predicted score vs target

![predicted vs psnr](figures/predicted_vs_psnr_2026-10-27_run01.png)

## Split

- train scenes (6): `bedroom`, `garage`, `kitchen`, `lobby`, `office`, `studio`
- test scenes (2): `company_room`, `hallway`

## Honest notes

- Only 1 test capture(s) are of type `video`, too few for a separate correlation, so that column is blank.
- Fewer than 3 test captures have a PSNR target, so the correlation is against damage severity (clean > light > medium > heavy), not PSNR.
- `lightgbm_features` ranks the test captures as well as the model (0.76 vs 0.67). On this test set the trained model is not yet earning its place.
- A test set of this size cannot separate methods whose intervals overlap; read the CI, not the point estimate.
