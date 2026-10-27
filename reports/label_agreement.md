# Label agreement

**Not produced yet — this report needs human review, which no script can stand in for.**

`scripts/07_check_label_agreement.py` writes this file, overwriting the text below.
It needs `data/labels/human_labels.csv`, which only exists once people have
reviewed captures in the review screen.

## How to produce it

```bash
capturecheck label data                         # 1. automatic labels
streamlit run capturecheck/review/app.py        # 2. review the queue
python scripts/07_check_label_agreement.py      # 3. this report
```

For the agreement figures to mean anything, **two reviewers must review the same
subset** — about 20 captures is enough. The script reads the first two reviewers
of each shared capture, so each person should enter their own name in the
sidebar and work through the same list.

## What it will contain

| Column | What it tells you |
|---|---|
| `precision` | Of the captures the rules flagged, how many did the reviewers agree were really bad |
| `recall` | Of the captures the reviewers called bad, how many the rules caught |
| `reviewer kappa` | How much the two reviewers agreed with each other, corrected for chance |

The kappa column is the ceiling: automatic labels cannot be more reliable than
the people they are measured against. A label with a low kappa is one people
cannot agree on, and the fix is to redefine it in
`capturecheck/labeling/schema.py`, not to retune its threshold.

The report ends with a list of labels below the precision bar and the exact
config key to change for each, for example:

> - **motion_blur** (precision 0.62, recall 0.95): too many false alarms: make
>   the threshold stricter. Tune `blur.sharp_min_laplacian_var /
>   blur.video_blur_ratio`.

## Why this matters

The thresholds in `configs/labeling.yaml` are starting values, taken from the
build guide and sanity-checked on synthetic captures. They have **not** been
calibrated against human judgement on real captures. Until this report exists,
every precision figure quoted for a problem label is measured against the
rules' own output, not against what a person would say — which tells you the
model reproduces the rules, not that the rules are right.
