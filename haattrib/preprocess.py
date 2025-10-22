import pandas as pd
def explode_paths(df: pd.DataFrame):
    # Expand "A>B>C" into rows with position for Markov transitions
    rows = []
    for _, r in df.iterrows():
        steps = r['path'].split('>')
        for i, ch in enumerate(steps):
            rows.append(dict(user=r.user, pos=i, channel=ch, convert=r.convert))
    return pd.DataFrame(rows)
