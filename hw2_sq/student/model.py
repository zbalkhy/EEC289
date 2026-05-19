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

        #self.lin = nn.Linear(in_dim, obs_dim)
        self.kinematic_scale = nn.Parameter(torch.tensor([1.0,0.9]))
        self.residual_scale = nn.Parameter(torch.tensor([0.05, 0.05, 1.0, 1.0]))

        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.head = nn.Linear(hidden_dim, obs_dim)
        
        # Initialize head weights to zero
        init.zeros_(self.head.weight)

        # Initialize head bias to zero
        if self.head.bias is not None:
            init.zeros_(self.head.bias)

        

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
        resid = self.head(feat)
        raw_delta = resid*self.residual_scale
        raw_delta[...,0] += self.kinematic_scale[0] * obs_norm[...,2]
        raw_delta[...,1] += self.kinematic_scale[0] * obs_norm[...,3]
        
        #raw_delta_linear = self.lin(torch.cat([obs_norm, act_norm], dim=-1))
        #raw_delta = raw_delta_linear + raw_delta_nl
        delta = raw_delta.clone()
        delta[..., 2:] = self.delta_limit * torch.tanh(raw_delta[..., 2:] / self.delta_limit)
        return delta, hidden
