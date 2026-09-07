"""Generate reverse and conservative low-bag corrections after public feedback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_jointstress_ensemble import shift_to_mean

ROOT = SRC_DIR.parent
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ANCHOR_OOF = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
LOWBAG_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_lowbag_oof.csv"
ANCHOR_TEST = SUBMISSION_DIR / "diverse_ordered20_w040_mean015.csv"
LOWBAG_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_lowbag_10fold_100.csv"
EXPECTED_PREVALENCE = 0.15


def main() -> None:
    anchor_oof = pd.read_csv(ANCHOR_OOF)
    lowbag_oof = pd.read_csv(LOWBAG_OOF)
    anchor_test = pd.read_csv(ANCHOR_TEST)
    lowbag_test = pd.read_csv(LOWBAG_TEST)
    assert anchor_oof["ID"].tolist() == lowbag_oof["ID"].tolist()
    assert anchor_test["ID"].tolist() == lowbag_test["ID"].tolist()
    anchor_oof_eta = logit(np.clip(anchor_oof["prediction"].to_numpy(), 1e-6, 1 - 1e-6))
    lowbag_oof_eta = logit(np.clip(lowbag_oof["prediction"].to_numpy(), 1e-6, 1 - 1e-6))
    anchor_test_eta = logit(np.clip(anchor_test["Target"].to_numpy(), 1e-6, 1 - 1e-6))
    lowbag_test_eta = logit(np.clip(lowbag_test["Target"].to_numpy(), 1e-6, 1 - 1e-6))
    results = {}
    for weight in [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        oof = shift_to_mean(anchor_oof_eta + weight * (lowbag_oof_eta - anchor_oof_eta), EXPECTED_PREVALENCE)[0]
        test = shift_to_mean(anchor_test_eta + weight * (lowbag_test_eta - anchor_test_eta), EXPECTED_PREVALENCE)[0]
        weight_label = f"{weight:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
        filename = f"lowbag_anchor_w{weight_label}.csv"
        submission = anchor_test.copy()
        submission["Target"] = np.clip(test, 1e-6, 1 - 1e-6)
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        results[filename] = {"weight": weight, "mean": float(test.mean()), "std": float(test.std()), "oof_mean": float(oof.mean())}
    report = {"anchor": ANCHOR_TEST.name, "weights": results}
    (ARTIFACT_DIR / "lowbag_reverse_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()