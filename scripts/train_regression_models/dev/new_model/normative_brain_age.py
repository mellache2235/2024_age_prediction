
# normative_brain_age.py
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ids: Optional[List[str]] = None):
        if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray): y = torch.tensor(y, dtype=torch.float32)
        assert X.ndim == 3 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self.X, self.y = X, y.view(-1)
        self.ids = ids if ids is not None else [str(i) for i in range(len(y))]
        assert len(self.ids) == len(self.y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.ids[idx]

def make_loader(X, y, ids=None, batch_size=32, shuffle=True, num_workers=0):
    ds = TimeSeriesDataset(X, y, ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding,
                                   dilation=dilation, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, channels: int, out_ch: int, kernels: Tuple[int, ...] = (3,5,7),
                 dilations: Tuple[int, ...] = (1,2,4), dropout: float = 0.3, groups_gn: int = 8):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv1d(channels, channels, k, d) for k,d in zip(kernels,dilations)
        ])
        self.merge = nn.Conv1d(channels*len(kernels), out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        self.act = nn.PReLU(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(channels, out_ch, kernel_size=1, bias=False) if channels!=out_ch else nn.Identity()
    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        y = self.dropout(self.act(self.norm(self.merge(y))))
        return y + self.res_proj(x)

class NormativeAgeNet(nn.Module):
    def __init__(self, in_ch: int = 246, stem_ch: int = 128, blocks: Tuple[int,int,int]=(128,64,64),
                 dropout: float = 0.3, embed_dim: int = 64):
        super().__init__()
        self.inst_norm = nn.InstanceNorm1d(in_ch, affine=False, eps=1e-5)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, stem_ch), num_channels=stem_ch),
            nn.PReLU(stem_ch),
        )
        chs = [stem_ch] + list(blocks)
        self.stages = nn.ModuleList([MultiScaleTemporalBlock(chs[i], chs[i+1], dropout=dropout) for i in range(len(chs)-1)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(chs[-1], embed_dim), nn.PReLU(embed_dim), nn.Dropout(dropout))
        self.mu_head = nn.Linear(embed_dim, 1)
        self.logvar_head = nn.Linear(embed_dim, 1)
    def forward_features(self, x):
        x = self.inst_norm(x); x = self.stem(x)
        for blk in self.stages: x = blk(x)
        x = self.dropout(x.mean(dim=2))
        return self.fc(x)
    def forward(self, x, return_features: bool=False):
        emb = self.forward_features(x)
        mu = self.mu_head(emb).squeeze(-1); log_var = self.logvar_head(emb).squeeze(-1)
        return (mu, log_var, emb) if return_features else (mu, log_var)

def heteroscedastic_gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    log_var = torch.clamp(log_var, min=-10.0, max=5.0)
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (y - mu) ** 2 * inv_var)

@torch.no_grad()
def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean(torch.abs(y_pred - y_true)).item()

from dataclasses import dataclass
@dataclass
class BiasParams:
    alpha: float
    beta: float

def fit_bias_correction(y_true: np.ndarray, y_pred: np.ndarray) -> BiasParams:
    X = np.vstack([np.ones_like(y_true), y_true]).T
    theta, _, _, _ = np.linalg.lstsq(X, y_pred, rcond=None)
    return BiasParams(alpha=float(theta[0]), beta=float(theta[1]))

def apply_bias_correction(y_true: np.ndarray, y_pred: np.ndarray, params: BiasParams) -> np.ndarray:
    beta = params.beta if abs(params.beta) > 1e-8 else 1.0
    return (y_pred - params.alpha) / beta

def compute_bag(y_true: np.ndarray, y_pred_corr: np.ndarray) -> np.ndarray:
    return y_pred_corr - y_true

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 1.0
    seed: int = 42

class NormativeTrainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.best_state = None; self.best_val_mae = float("inf"); self.epochs_no_improve = 0
    def step(self, batch):
        x,y,_ = batch; x=x.to(self.cfg.device); y=y.to(self.cfg.device)
        mu, log_var = self.model(x)
        loss = heteroscedastic_gaussian_nll(mu, log_var, y).mean()
        return loss, mu.detach()
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval(); all_mu, all_y = [], []
        for x,y,_ in loader:
            x=x.to(self.cfg.device); y=y.to(self.cfg.device)
            mu,_ = self.model(x); all_mu.append(mu); all_y.append(y)
        mu = torch.cat(all_mu); y = torch.cat(all_y)
        return {"mae": mae(mu, y)}
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        torch.cuda.empty_cache(); set_seed(self.cfg.seed)
        for epoch in range(1, self.cfg.epochs+1):
            self.model.train(); epoch_loss = 0.0
            for batch in train_loader:
                self.opt.zero_grad(set_to_none=True)
                loss,_ = self.step(batch); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step(); epoch_loss += loss.item()
            val_metrics = {"mae": float("nan")}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                if val_metrics["mae"] < self.best_val_mae - 1e-6:
                    self.best_val_mae = val_metrics["mae"]
                    self.best_state = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.cfg.patience: break
            print(f"Epoch {epoch:03d} | TrainNLL: {epoch_loss:.3f} | Val MAE: {val_metrics['mae']:.4f}")
        if self.best_state is not None: self.model.load_state_dict(self.best_state)
    @torch.no_grad()
    def predict(self, loader: DataLoader, return_features: bool = False):
        self.model.eval(); mus, logvars, ys, ids, embs = [], [], [], [], []
        for x,y,sid in loader:
            x=x.to(self.cfg.device)
            if return_features:
                mu, log_var, emb = self.model(x, return_features=True); embs.append(emb.detach().cpu())
            else:
                mu, log_var = self.model(x)
            mus.append(mu.detach().cpu()); logvars.append(log_var.detach().cpu()); ys.append(y.detach().cpu()); ids.extend(sid)
        mu = torch.cat(mus).numpy(); log_var = torch.cat(logvars).numpy(); y = torch.cat(ys).numpy()
        if return_features:
            emb = torch.cat(embs).numpy(); return mu, log_var, y, ids, emb
        return mu, log_var, y, ids

def save_predictions_csv(path: str, ids: List[str], age_true: np.ndarray, age_pred: np.ndarray,
                         age_pred_bc: np.ndarray, bag: np.ndarray, extra: Optional[Dict[str, np.ndarray]] = None):
    import pandas as pd
    df = {"subject_id": ids, "age_true": age_true, "age_pred": age_pred,
          "age_pred_bias_corrected": age_pred_bc, "brain_age_gap": bag}
    if extra is not None:
        for k,v in extra.items(): df[k]=v
    df = pd.DataFrame(df); df.to_csv(path, index=False); return df
