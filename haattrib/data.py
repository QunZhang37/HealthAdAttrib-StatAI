import numpy as np, pandas as pd

CHANNELS = ["Search","Social","Email","Display","Referral","Video","Direct","CallCenter"]
TERMINALS = ["Start","Convert","Null"]

def simulate_journeys(n_users=10000, seed=42, max_len=8):
    rng = np.random.default_rng(seed)
    users = np.arange(n_users)
    # Base channel propensities (healthcare context)
    base_p = np.array([0.22,0.14,0.18,0.10,0.08,0.07,0.15,0.06])
    base_p = base_p/base_p.sum()

    journeys = []
    for u in users:
        L = rng.integers(2, max_len+1)
        path = list(rng.choice(CHANNELS, size=L, p=base_p))
        # Conversion probability shaped by path & demographics
        age = rng.integers(18,85)
        chronic = rng.binomial(1, 0.25 + 0.002*max(age-45,0))
        # path effect: Search→CallCenter or Email→Direct increases odds
        boost = 0.0
        s = "->".join(path)
        if "Search->CallCenter" in s: boost += 0.6
        if "Email->Direct" in s: boost += 0.4
        if "Referral" in path: boost += 0.3
        if "Video" in path and "Social" in path: boost += 0.2
        logit = -1.2 + 0.01*(age-50) + 0.5*chronic + boost + 0.1*len(set(path))
        p_conv = 1/(1+np.exp(-logit))
        convert = rng.binomial(1, p_conv)
        journeys.append(dict(user=u, age=age, chronic=chronic, path=">".join(path), convert=convert))
    return pd.DataFrame(journeys)
