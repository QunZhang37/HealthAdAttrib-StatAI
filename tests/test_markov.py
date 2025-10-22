import pandas as pd
from haattrib.data import simulate_journeys, CHANNELS
from haattrib.models.markov import removal_effect

def test_removal_effect_runs():
    df = simulate_journeys(n_users=500, seed=1)
    contrib = removal_effect(df, CHANNELS)
    assert abs(sum(contrib.values()) - 1.0) < 1e-6
