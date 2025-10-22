import torch, torch.nn as nn
import numpy as np

CHANNELS = ['Search','Social','Email','Display','Referral','Video','Direct','CallCenter']

class LSTMAttrib(nn.Module):
    def __init__(self, emb_dim=16, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(len(CHANNELS), emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out[:,-1,:])
        return logits.squeeze(-1)

def encode_paths(paths):
    ch2i = {c:i for i,c in enumerate(CHANNELS)}
    seqs = []
    for p in paths:
        idxs = [ch2i[c] for c in p.split('>') if c in ch2i]
        seqs.append(idxs)
    maxlen = max(len(s) for s in seqs)
    arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i,s in enumerate(seqs):
        arr[i, :len(s)] = s
    return arr

def train_lstm(paths, y, epochs=5, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    X = encode_paths(paths)
    y = torch.tensor(y).float()
    model = LSTMAttrib()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        xb = torch.tensor(X); yb = y
        opt.zero_grad()
        logits = model(xb)
        loss = bce(logits, yb)
        loss.backward(); opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(X))).numpy()
    return model, prob
