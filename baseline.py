"""The 7-h-5 ReLU6 MLP used for both stages, and its training loop (Adam, early stopping on held-out
training subjects). Trimmed from the research repo's stage-1 baseline script."""
import numpy as np
import torch

torch.set_num_threads(6)


class MLP(torch.nn.Module):
    def __init__(self, n_in=7, hidden=6):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(n_in, hidden), torch.nn.Linear(hidden, 5)

    def forward(self, x):
        return self.l2(torch.clamp(self.l1(x), 0, 6))  # ReLU6


def train(Xtr, ytr, Xva, yva, seed, hidden=6):
    torch.manual_seed(seed)
    m, best, bad, state = MLP(Xtr.shape[1], hidden), np.inf, 0, None
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(400):
        m.train()
        for idx in torch.randperm(len(Xt)).split(64):
            opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = lossf(m(Xv), yv).item()
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        elif (bad := bad + 1) >= 30:
            break
    m.load_state_dict(state)
    return m
