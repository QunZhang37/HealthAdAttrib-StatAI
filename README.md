# HealthAdAttrib-StatAI: Statistical Attribution Modeling for Multi-Channel Healthcare Advertising

**Short description:** Research-grade framework for *probabilistic and AI-enhanced attribution modeling* across multi-channel healthcare campaigns. Implements **Markov-chain removal effects**, **(Bayesian) logistic attribution**, **Shapley-value path attribution**, and **deep sequence models (LSTM)**; includes simulation, evaluation (incremental lift & ROI), visualization, and a **Streamlit dashboard**.

## Key features
- **Synthetic EHR-safe marketing dataset generator** (privacy-preserving): touchpoints → conversions
- **Attribution models**: Markov removal, logistic/Bayesian logit, Shapley path values, deep LSTM
- **Evaluation**: conversion AUC, calibration, channel contribution stability, counterfactual path tests
- **Visualization**: channel transition graphs, attribution bar charts, path distributions
- **Reproducibility**: deterministic seeds, config-driven scripts, tests

> ⚠️ Use real patient marketing data only under HIPAA/IRB-compliant processes.

---

## Quickstart
```bash
# Setup
pip install -r requirements.txt
# or conda
# conda env create -f environment.yml && conda activate haattrib

# 1) Generate synthetic journeys
python -m haattrib.scripts.make_dataset --n_users 20000 --seed 42 --out data/processed/journeys.csv

# 2) Run full pipeline (fit Markov/logit/Shapley/LSTM; export metrics + plots)
python -m haattrib.scripts.run_all --data data/processed/journeys.csv --out outputs/run1

# 3) Launch dashboard
streamlit run haattrib/dashboard/app.py
```

## Repo layout
```
HealthAdAttrib-StatAI/
 ├─ haattrib/
 │  ├─ __init__.py
 │  ├─ utils.py
 │  ├─ data.py                 # synthetic generator
 │  ├─ preprocess.py           # sessionization, sequence encoding
 │  ├─ viz.py                  # graphs & attribution plots
 │  ├─ eval.py                 # metrics, ROI, stability
 │  ├─ models/
 │  │   ├─ logistic.py         # sklearn logistic + optional PyMC Bayesian
 │  │   ├─ markov.py           # transition matrix & removal effects
 │  │   ├─ shapley.py          # Monte Carlo path Shapley
 │  │   └─ deepseq.py          # PyTorch LSTM sequence model
 │  ├─ dashboard/app.py        # Streamlit dashboard
 │  └─ scripts/
 │      ├─ make_dataset.py
 │      └─ run_all.py
 ├─ docs/
 │  ├─ paper-outline.md
 │  └─ paper-latex/main.tex
 ├─ tests/
 │  ├─ test_sim.py
 │  └─ test_markov.py
 ├─ data/README.md
 ├─ .gitignore
 ├─ LICENSE
 ├─ CITATION.cff
 ├─ CODE_OF_CONDUCT.md
 ├─ CONTRIBUTING.md
 ├─ SECURITY.md
 ├─ requirements.txt
 ├─ environment.yml
 ├─ pyproject.toml
 ├─ setup.cfg
 ├─ Makefile
 └─ Dockerfile
```

## Notes
- Streamlit dashboard expects artifacts under `outputs/<run>/` (metrics, plots, JSON attribution).
- Shapley calculation uses **Monte Carlo sampling** for scalability.
- Bayesian logistic regression (PyMC) is optional; falls back to frequentist if PyMC not installed.
