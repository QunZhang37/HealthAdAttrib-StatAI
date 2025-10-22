import argparse, os, json, pandas as pd
from haattrib.models.markov import removal_effect
from haattrib.models.shapley import shapley
from haattrib.models.logistic import fit_logit
from haattrib.models.deepseq import train_lstm
from haattrib.viz import plot_channel_contributions
from haattrib.eval import binary_metrics
from haattrib.data import CHANNELS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)

    # Logistic
    logit_model, logit_prob = fit_logit(df)
    logit_metrics = binary_metrics(df.convert.values, logit_prob)

    # Markov removal effects
    markov_contrib = removal_effect(df, CHANNELS)
    plot_channel_contributions(markov_contrib, os.path.join(args.out, "markov_contrib.png"))

    # Shapley (MC)
    shapley_contrib = shapley(df, CHANNELS, n_samples=1000, seed=42)
    plot_channel_contributions(shapley_contrib, os.path.join(args.out, "shapley_contrib.png"))

    # Deep LSTM
    lstm_model, lstm_prob = train_lstm(df.path.values.tolist(), df.convert.values, epochs=3)
    lstm_metrics = binary_metrics(df.convert.values, lstm_prob)

    json.dump({
        "logit_metrics": logit_metrics,
        "lstm_metrics": lstm_metrics,
        "markov_contrib": markov_contrib,
        "shapley_contrib": shapley_contrib
    }, open(os.path.join(args.out, "summary.json"), "w"), indent=2)

    print("Run complete. Artifacts written to", args.out)
