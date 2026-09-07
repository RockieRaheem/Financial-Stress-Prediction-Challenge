"""Build the OOF-selected low-bag correction around the public anchor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_jointstress_ensemble import shift_to_mean

ROOT = SRC_DIR.parent
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
ANCHOR_OOF = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
LOWBAG_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_lowbag_oof.csv"
ANCHOR_TEST = SUBMISSION_DIR / "diverse_ordered20_w040_mean015.csv"
LOWBAG_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_lowbag_10fold_100.csv"
OUTPUT = SUBMISSION_DIR / "lowbag_anchor_w015_mean015.csv"


def main() -> None:
    anchor_oof = pd.read_csv(ANCHOR_OOF)
    lowbag_oof = pd.read_csv(LOWBAG_OOF)
    anchor_test = pd.read_csv(ANCHOR_TEST)
    lowbag_test = pd.read_csv(LOWBAG_TEST)
    assert anchor_oof["ID"].tolist() == lowbag_oof["ID"].tolist()
    assert anchor_test["ID"].tolist() == lowbag_test["ID"].tolist()
    weight = 0.15
    oof_eta = logit(np.clip(anchor_oof["prediction"], 1e-6, 1 - 1e-6))
    lowbag_oof_eta = logit(np.clip(lowbag_oof["prediction"], 1e-6, 1 - 1e-6))
    test_eta = logit(np.clip(anchor_test["Target"], 1e-6, 1 - 1e-6))
    lowbag_test_eta = logit(np.clip(lowbag_test["Target"], 1e-6, 1 - 1e-6))
    oof = shift_to_mean(oof_eta + weight * (lowbag_oof_eta - oof_eta), 0.15)[0]
    test = shift_to_mean(test_eta + weight * (lowbag_test_eta - test_eta), 0.15)[0]
    output = anchor_test.copy()
    output["Target"] = np.clip(test, 1e-6, 1 - 1e-6)
    output.to_csv(OUTPUT, index=False)
    print({"output": OUTPUT.name, "rows": len(output), "mean": float(test.mean()), "std": float(test.std()), "oof_mean": float(oof.mean())})


if __name__ == "__main__":
    main()