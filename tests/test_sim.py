from haattrib.data import simulate_journeys
def test_sim_shapes():
    df = simulate_journeys(n_users=200, seed=0)
    assert len(df) == 200
    for c in ['user','age','chronic','path','convert']:
        assert c in df.columns
