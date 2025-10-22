import numpy as np, pandas as pd
from collections import Counter, defaultdict

def build_transition_matrix(df: pd.DataFrame, channels):
    # Add Start/Convert/Null terminals
    states = ['Start'] + channels + ['Convert','Null']
    idx = {s:i for i,s in enumerate(states)}
    T = np.zeros((len(states), len(states)), dtype=float)
    for _, r in df.iterrows():
        steps = r['path'].split('>')
        seq = ['Start'] + steps + (['Convert'] if r['convert']==1 else ['Null'])
        for a,b in zip(seq[:-1], seq[1:]):
            T[idx[a], idx[b]] += 1
    # Row normalize
    T = T / np.maximum(T.sum(1, keepdims=True), 1e-12)
    return states, T

def absorption_prob(states, T):
    # Probability of reaching Convert from Start
    s_idx = states.index('Start')
    c_idx = states.index('Convert')
    # Power iteration style simulation
    p = np.zeros(len(states)); p[s_idx] = 1.0
    for _ in range(50):
        p = p @ T
    return float(p[c_idx])

def removal_effect(df: pd.DataFrame, channels):
    states, T = build_transition_matrix(df, channels)
    base = absorption_prob(states, T)
    contrib = {}
    for ch in channels:
        # Remove channel: zero out transitions to/from it, renormalize remaining rows
        s_idx = states.index(ch)
        T_removed = T.copy()
        T_removed[:, s_idx] = 0.0; T_removed[s_idx, :] = 0.0
        # renormalize rows (excluding all-zero rows)
        row_sums = T_removed.sum(1, keepdims=True)
        T_removed = T_removed / np.maximum(row_sums, 1e-12)
        alt = absorption_prob(states, T_removed)
        contrib[ch] = max(base - alt, 0.0)
    # Normalize to sum to 1
    s = sum(contrib.values()) + 1e-12
    return {k: v/s for k,v in contrib.items()}
