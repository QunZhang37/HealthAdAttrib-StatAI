import matplotlib.pyplot as plt, networkx as nx, pandas as pd, numpy as np, os

def plot_channel_contributions(contrib: dict, out_path: str):
    labels = list(contrib.keys()); vals = [contrib[k] for k in labels]
    plt.figure(figsize=(8,4))
    idx = np.argsort(vals)[::-1]
    plt.bar(np.array(labels)[idx], np.array(vals)[idx])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Attribution Share')
    plt.title('Channel Contribution')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
