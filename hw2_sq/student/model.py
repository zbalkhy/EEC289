"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.init as init


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = False,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        in_dim = obs_dim + act_dim

        self.lin = nn.Linear(in_dim, 2)
        self.kinematic_scale = nn.Parameter(torch.tensor([1.0,0.9]))
        self.accel_residual_scale = nn.Parameter(torch.ones(2))

        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.head = nn.Linear(hidden_dim, 2)
        
        # Initialize head weights to zero
        init.zeros_(self.head.weight)
        init.zeros_(self.lin.weight)

        # Initialize head bias to zero
        if self.head.bias is not None:
            init.zeros_(self.head.bias)
            init.zeros_(self.lin.bias)

        

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        obs_act = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.encoder(obs_act)
        feat = self.layer_norm(feat)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        accel_raw = self.lin(obs_act) + self.head(feat) * self.accel_residual_scale
        accel_delta = self.delta_limit * torch.tanh(accel_raw / self.delta_limit)

        velocity = torch.stack([obs_norm[..., 2], obs_norm[..., 3]], dim=-1)
        pos_delta = self.kinematic_scale * velocity
        delta = torch.cat([pos_delta, accel_delta], dim=-1)
        return delta, hidden
