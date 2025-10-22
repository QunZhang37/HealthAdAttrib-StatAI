import argparse, os, pandas as pd
from haattrib.data import simulate_journeys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/processed/journeys.csv")
    args = ap.parse_args()
    df = simulate_journeys(n_users=args.n_users, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")
