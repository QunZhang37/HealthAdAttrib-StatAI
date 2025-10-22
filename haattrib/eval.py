import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def binary_metrics(y_true, y_prob):
    return dict(
        auc = float(roc_auc_score(y_true, y_prob)),
        brier = float(brier_score_loss(y_true, y_prob))
    )

def roi(contrib: dict, cost: dict, value_per_conv=1000.0):
    # Simple expected ROI by channel: value share - cost
    total_share = sum(contrib.values()) + 1e-12
    roi = {}
    for ch, share in contrib.items():
        expected_value = value_per_conv * (share/total_share)
        roi[ch] = expected_value - cost.get(ch, 0.0)
    return roi
