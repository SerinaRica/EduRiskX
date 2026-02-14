import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .base_predictor import BasePredictor

class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(patch_dim, d_model)
    def forward(self, x):
        return self.proj(x)

class PatchTSTPredictor(BasePredictor):
    def __init__(self, input_dim, max_len, patch_size=4, stride=4, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        self.input_dim = input_dim
        self.max_len = max_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.patch_embed = PatchEmbedding(patch_size*input_dim, d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.registered = False
    def _num_patches(self, length):
        if length <= 0:
            return 1
        if length < self.patch_size:
            return 1
        return 1 + (max(0, length - self.patch_size)) // self.stride
    def _make_patches(self, x, lengths):
        B, T, D = x.size()
        max_patches = self._num_patches(T)
        patches = []
        lens = []
        for i in range(B):
            L = int(lengths[i].item()) if lengths is not None else T
            P = self._num_patches(L)
            seq = x[i]
            ps = []
            start = 0
            count = 0
            while count < P:
                end = min(start + self.patch_size, T)
                pad_len = self.patch_size - (end - start)
                patch = seq[start:end]
                if pad_len > 0:
                    pad = torch.zeros(pad_len, D, device=seq.device)
                    patch = torch.cat([patch, pad], dim=0)
                ps.append(patch.reshape(-1))
                start += self.stride
                count += 1
            while len(ps) < max_patches:
                ps.append(torch.zeros(self.patch_size*D, device=seq.device))
            patches.append(torch.stack(ps, dim=0))
            lens.append(P)
        patches = torch.stack(patches, dim=0)
        lens = torch.tensor(lens, device=x.device, dtype=torch.long)
        return patches, lens
    def _forward(self, x, lengths):
        patches, plens = self._make_patches(x, lengths)
        emb = self.patch_embed(patches)
        out = self.encoder(emb)
        mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) < plens.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        logits = self.fc(pooled)
        return logits.squeeze(1), pooled
    def _to_device(self, device):
        self.patch_embed.to(device)
        self.encoder.to(device)
        self.fc.to(device)
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=3, batch_size=64, lr=1e-3, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._to_device(device)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        lengths = torch.tensor(np.maximum(1, np.sum((X_train!=0).any(axis=2), axis=1)), dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t, lengths), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(list(self.patch_embed.parameters())+list(self.encoder.parameters())+list(self.fc.parameters()), lr=lr)
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
