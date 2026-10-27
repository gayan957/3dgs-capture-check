# Evaluation report

Run `2026-10-27_run01` · 17 unseen test captures (1 video, 16 image bundle; 2 with a PSNR target)

Test scenes were never trained on, and scenes marked `company` are test-only.

## Ranking quality

Spearman correlation between each method's score and the quality target, with a percentile bootstrap 95% interval.

| method | Spearman (95% CI) | videos | image bundles |
|---|---|---|---|
| **capturecheck** | 0.69 (0.24 – 0.94) | n/a | 0.69 |
| `frame_count` | 0.31 (-0.19 – 0.73) | n/a | 0.27 |
| `mean_blur` | -0.08 (-0.58 – 0.43) | n/a | -0.01 |
| `orb_matches` | 0.59 (0.14 – 0.91) | n/a | 0.51 |
| `lightgbm_features` | 0.75 (0.40 – 0.94) | n/a | 0.69 |

## Problems found

Precision and recall of the model's problem heads against the labels.

| problem | test captures | positives | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `motion_blur` | 17 | 3 | 2 | 1.00 | 0.67 | 0.80 |
| `lighting` | 17 | 3 | 3 | 1.00 | 1.00 | 1.00 |
| `no_walking` | 17 | 0 | 1 | 0.00 | n/a | n/a |
| `poor_coverage` | 17 | 16 | 16 | 1.00 | 1.00 | 1.00 |
| `too_few_views` | 17 | 3 | 11 | 0.27 | 1.00 | 0.43 |
| `low_overlap` | 17 | 6 | 6 | 1.00 | 1.00 | 1.00 |
| `mixed_cameras` | 17 | 0 | 0 | n/a | n/a | n/a |
| `reflective` | 17 | 0 | 0 | n/a | n/a | n/a |
| `low_texture` | 17 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| `noisy` | 17 | 3 | 3 | 1.00 | 1.00 | 1.00 |

## Runtime

- Measuring a capture: **2.56 s** (OpenCV checks + DINOv2, mean over everything labelled)
- Scoring it with the model: **0.0038 s** (onnx)
- Budget: 60 s per capture on a laptop CPU

## Predicted score vs target

![predicted vs psnr](figures/predicted_vs_psnr_2026-10-27_run01.png)

## Split

- train scenes (6): `bedroom`, `garage`, `kitchen`, `lobby`, `office`, `studio`
- test scenes (2): `company_room`, `hallway`

## Honest notes

- Only 1 test capture(s) are of type `video`, too few for a separate correlation, so that column is blank.
- Fewer than 3 test captures have a PSNR target, so the correlation is against damage severity (clean > light > medium > heavy), not PSNR.
- `lightgbm_features` ranks the test captures as well as the model (0.75 vs 0.69). On this test set the trained model is not yet earning its place.
- A test set of this size cannot separate methods whose intervals overlap; read the CI, not the point estimate.
