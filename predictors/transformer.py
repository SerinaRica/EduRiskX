import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .base_predictor import BasePredictor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        # Weighted sum
        context = torch.matmul(attn, v)
        return context + x  # Residual connection

class TransformerPredictor(BasePredictor, nn.Module):
    def __init__(self, input_dim, max_len, d_model=128, nhead=8, num_layers=3, dropout=0.2, use_layernorm=True, use_class_weight=True, use_temporal_attn=True):
        nn.Module.__init__(self) # Initialize nn.Module
        self.input_dim = input_dim
        self.max_len = max_len
        self.d_model = d_model
        self.use_layernorm = use_layernorm
        self.use_class_weight = use_class_weight
        self.use_temporal_attn = use_temporal_attn
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.model = nn.Sequential()
        self.proj = nn.Linear(input_dim, d_model)
        if self.use_layernorm:
            self.ln_input = nn.LayerNorm(d_model) # Add LayerNorm after projection
        self.pe = PositionalEncoding(d_model, max_len)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        if self.use_temporal_attn:
            self.temporal_attn = TemporalAttention(d_model)
            self.ln_attn = nn.LayerNorm(d_model)
            
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def _forward(self, x, lengths):
        x = self.proj(x)
        if self.use_layernorm:
            x = self.ln_input(x) # Normalize projected inputs
        x = self.pe(x)
        x = self.encoder(x)
        
        if self.use_temporal_attn:
            attn_out = self.temporal_attn(x)
            x = self.ln_attn(attn_out)
        
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        else:
            pooled = x.mean(dim=1)
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits.squeeze(1), pooled
    
    # Forward method required for nn.Module if used directly, but we use _forward internaly or via predict_proba
    def forward(self, x, lengths=None):
         return self._forward(x, lengths)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=64, lr=1e-3, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        lengths = torch.tensor(np.maximum(1, np.sum((X_train!=0).any(axis=2), axis=1)), dtype=torch.long)
        
        if X_val is not None and y_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.float32)
            lengths_v = torch.tensor(np.maximum(1, np.sum((X_val!=0).any(axis=2), axis=1)), dtype=torch.long)
            val_dataset = TensorDataset(X_v, y_v, lengths_v)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Calculate class weight for imbalance
        if self.use_class_weight:
            num_pos = y_t.sum()
            num_neg = len(y_t) - num_pos
            pos_weight = num_neg / (num_pos + 1e-5)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            pos_weight = torch.tensor(1.0)
        
        dataset = TensorDataset(X_t, y_t, lengths)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Use AdamW and Scheduler
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
        
        print(f"Training Transformer: d_model={self.d_model}, layers={len(self.encoder.layers)}, pos_weight={pos_weight.item():.2f}, LN={self.use_layernorm}")
        
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.train() # Ensure train mode
            total_loss = 0
            for xb, yb, lb in loader:
                xb = xb.to(device); yb = yb.to(device); lb = lb.to(device)
                opt.zero_grad()
                logits, _ = self._forward(xb, lb)
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) # Gradient clipping
                opt.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss/len(loader)
            history['loss'].append(avg_loss)
            
            # Validation
            if val_loader:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb, lb in val_loader:
                        xb = xb.to(device); yb = yb.to(device); lb = lb.to(device)
                        logits, _ = self._forward(xb, lb)
                        loss = loss_fn(logits, yb)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
            
            # Optional: Print loss every few epochs
            if (epoch + 1) % 5 == 0:
                val_msg = f", Val Loss: {history['val_loss'][-1]:.4f}" if val_loader else ""
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}{val_msg}")
        
        return history

    def predict_proba(self, X, lengths=None, batch_size=128, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval() # Ensure eval mode
        
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
