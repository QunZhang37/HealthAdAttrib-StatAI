import numpy as np, random, os, json
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)
