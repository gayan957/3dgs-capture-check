"""Fit LightGBM in a clean interpreter and write the predictions out.

Run as a subprocess, never imported by the rest of the package. On macOS,
LightGBM and PyTorch each ship their own copy of ``libomp``, and loading both
into one process segfaults; a separate interpreter that never imports torch has
only one copy. See :func:`capturecheck.model.baselines.lightgbm_baseline`.

Usage: ``python -I -m capturecheck.model._lightgbm_worker <input.npz> <output.npy>``
"""

from __future__ import annotations

import sys

import numpy as np


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    with np.load(argv[1], allow_pickle=False) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        seed = int(data["seed"])

    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    np.save(argv[2], np.asarray(model.predict(x_test), dtype=np.float64))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
