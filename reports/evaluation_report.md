# Evaluation report

Run `2026-11-16_run01` · 35 unseen test captures (1 video, 34 image bundle; 0 with a PSNR target)

Test scenes were never trained on, and scenes marked `company` are test-only.

## Ranking quality

Spearman correlation between each method's score and the quality target, with a percentile bootstrap 95% interval.

| method | Spearman (95% CI) | videos | image bundles |
|---|---|---|---|
| **capturecheck** | 0.46 (0.12 – 0.72) | n/a | 0.45 |
| `frame_count` | 0.20 (-0.15 – 0.52) | n/a | 0.20 |
| `mean_blur` | -0.04 (-0.38 – 0.35) | n/a | 0.01 |
| `orb_matches` | 0.27 (-0.07 – 0.57) | n/a | 0.21 |
| `lightgbm_features` | 0.66 (0.41 – 0.83) | n/a | 0.63 |

## Problems found

Precision and recall of the model's problem heads against the labels.

| problem | test captures | positives | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `motion_blur` | 35 | 14 | 14 | 0.86 | 0.86 | 0.86 |
| `lighting` | 35 | 35 | 35 | 1.00 | 1.00 | 1.00 |
| `no_walking` | 35 | 0 | 0 | n/a | n/a | n/a |
| `poor_coverage` | 35 | 15 | 14 | 0.57 | 0.53 | 0.55 |
| `too_few_views` | 35 | 6 | 2 | 1.00 | 0.33 | 0.50 |
| `low_overlap` | 35 | 35 | 35 | 1.00 | 1.00 | 1.00 |
| `mixed_cameras` | 35 | 0 | 0 | n/a | n/a | n/a |
| `reflective` | 35 | 0 | 0 | n/a | n/a | n/a |
| `low_texture` | 35 | 0 | 0 | n/a | n/a | n/a |
| `noisy` | 35 | 6 | 9 | 0.56 | 0.83 | 0.67 |

## Runtime

- Measuring a capture: **7.61 s** (OpenCV checks + DINOv2, mean over everything labelled)
- Scoring it with the model: **0.0062 s** (torch)
- Budget: 60 s per capture on a laptop CPU

## Predicted score vs target

![predicted vs psnr](figures/predicted_vs_psnr_2026-11-16_run01.png)

## Split

- train scenes (9): `Robotic lab`, `courthouse`, `flightroom`, `indoor`, `room_01`, `studyarea`, `village_01`, `washroom_01`, `water_bottle`
- test scenes (2): `nyc-apartment`, `studioroom`

## Honest notes

- Only 1 test capture(s) are of type `video`, too few for a separate correlation, so that column is blank.
- Fewer than 3 test captures have a PSNR target, so the correlation is against damage severity (clean > light > medium > heavy), not PSNR.
- `lightgbm_features` ranks the test captures as well as the model (0.66 vs 0.46). On this test set the trained model is not yet earning its place.
- A test set of this size cannot separate methods whose intervals overlap; read the CI, not the point estimate.
