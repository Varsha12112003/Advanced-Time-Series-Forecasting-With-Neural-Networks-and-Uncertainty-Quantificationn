import torch
import torch.nn as nn

class LSTMForecastMC(nn.Module):
    """
    LSTM encoder-decoder style model for multi-step forecasting with Monte Carlo dropout support.
    - input_size: number of features
    - hidden_size: LSTM hidden units
    - num_layers: LSTM layers
    - dropout: dropout probability (applied between layers and for MC dropout in eval)
    - horizon: forecasting horizon (output steps)
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, horizon=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # A dropout module that we can keep active at inference for MC Dropout
        self.dropout = nn.Dropout(dropout)
        # simple linear head to forecast all horizon steps (flattened)
        self.head = nn.Linear(hidden_size, horizon)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        # use last time-step
        last = out[:, -1, :]
        dropped = self.dropout(last)
        preds = self.head(dropped)  # (batch, horizon)
        return preds

# helper to do MC sampling
def mc_predict(model, x, n_samples=100, device='cpu'):
    model.eval()
    # enable dropout during eval for MC Dropout
    # PyTorch Dropout modules are only active in train(); we can override by temporarily enabling train mode for dropout only
    # Simpler: call model.train() and keep grads off
    preds = []
    with torch.no_grad():
        model.train()  # keep dropout active
        for _ in range(n_samples):
            p = model(x.to(device))
            preds.append(p.cpu().numpy())
    import numpy as np
    preds = np.stack(preds, axis=0)  # (n_samples, batch, horizon)
    mean = preds.mean(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)
    return mean, lower, upper
