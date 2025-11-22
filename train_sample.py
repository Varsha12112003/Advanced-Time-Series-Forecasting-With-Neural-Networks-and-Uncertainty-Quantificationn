import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import LSTMForecastMC, mc_predict

# Create toy data (sine + noise) for demonstration
def create_toy_data(n_series=1000, seq_len=50, horizon=10):
    X = []
    Y = []
    for i in range(n_series):
        t = np.linspace(0, 20, seq_len + horizon) + np.random.randn() * 0.1
        series = np.sin(t) + 0.2*np.random.randn(len(t))
        X.append(series[:seq_len].reshape(-1,1))
        Y.append(series[seq_len:seq_len+horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X, Y = create_toy_data()
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = 'cpu'
model = LSTMForecastMC(input_size=1, hidden_size=64, dropout=0.2, horizon=Y.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# quick training loop
for epoch in range(3):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()*xb.size(0)
    print(f"Epoch {epoch} loss: {total / len(dataset):.4f}")

# MC predict on a small batch
xb, yb = next(iter(loader))
mean, lower, upper = mc_predict(model, xb[:8], n_samples=50, device=device)
print('mean shape', mean.shape)
