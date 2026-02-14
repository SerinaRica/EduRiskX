import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .base_predictor import BasePredictor

class ChannelPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_channels):
        super().__init__()
        pe = torch.zeros(num_channels, d_model)
        position = torch.arange(0, num_channels).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class ITransformerPredictor(BasePredictor):
    def __init__(self, input_dim, max_len, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        self.input_dim = input_dim
        self.max_len = max_len
        self.d_model = d_model
        self.channel_embed = nn.Linear(max_len, d_model)
        self.pe = ChannelPositionalEncoding(d_model, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def _forward(self, x, lengths):
        B, T, D = x.size()
        x = x.transpose(1, 2)
        x = self.channel_embed(x)
        x = self.pe(x)
        out = self.encoder(x)
        pooled = out.mean(dim=1)
        logits = self.fc(pooled)
        return logits.squeeze(1), pooled
    def _to_device(self, device):
        self.channel_embed.to(device)
        self.pe.to(device)
        self.encoder.to(device)
        self.fc.to(device)
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=3, batch_size=64, lr=1e-3, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._to_device(device)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        lengths = torch.tensor(np.maximum(1, np.sum((X_train!=0).any(axis=2), axis=1)), dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t, lengths), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(list(self.channel_embed.parameters())+list(self.pe.parameters())+list(self.encoder.parameters())+list(self.fc.parameters()), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            for xb, yb, lb in loader:
                xb = xb.to(device); yb = yb.to(device); lb = lb.to(device)
                opt.zero_grad()
                logits, _ = self._forward(xb, lb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
    def predict_proba(self, X, lengths=None, batch_size=128, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._to_device(device)
        X_t = torch.tensor(X, dtype=torch.float32)
        if lengths is None:
            lengths = torch.tensor(np.maximum(1, np.sum((X!=0).any(axis=2), axis=1)), dtype=torch.long)
        else:
            lengths = torch.tensor(lengths, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, lengths), batch_size=batch_size, shuffle=False)
        probs = []
        hiddens = []
        with torch.no_grad():
            for xb, lb in loader:
                xb = xb.to(device); lb = lb.to(device)
                logits, pooled = self._forward(xb, lb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.extend(p.tolist())
                hiddens.append(pooled.detach().cpu().numpy())
        hidden = np.concatenate(hiddens, axis=0)
        return np.array(probs), hidden
