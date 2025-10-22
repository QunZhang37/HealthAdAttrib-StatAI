import numpy as np, pandas as pd, random
from collections import defaultdict

def value_function(df: pd.DataFrame, coalition):
    # Value: conversion rate when only channels in coalition are allowed
    allowed = set(coalition)
    mask = df.path.apply(lambda p: all(ch in allowed for ch in set(p.split('>'))))
    if mask.sum()==0: return 0.0
    return df.loc[mask, 'convert'].mean()

def shapley(df: pd.DataFrame, channels, n_samples=2000, seed=42):
    random.seed(seed)
    phi = defaultdict(float)
    for _ in range(n_samples):
        perm = channels.copy(); random.shuffle(perm)
        coalition = []
        prev_val = 0.0
        for ch in perm:
            new_coal = coalition + [ch]
            v = value_function(df, new_coal)
            phi[ch] += (v - prev_val)
            coalition = new_coal; prev_val = v
    for ch in channels:
        phi[ch] /= n_samples
    # Normalize positive contributions
    s = sum(max(v,0.0) for v in phi.values()) + 1e-12
    return {k: max(v,0.0)/s for k,v in phi.items()}
