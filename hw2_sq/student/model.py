"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
from torch import nn


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
        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.head = nn.Linear(hidden_dim, obs_dim)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def calc_angular_velocity_delta(self, m1, m2, l, k_t, theta, F, g):
        total_mass = m1+m2
        mass_length = m2*l

        return (mass_length*F + total_mass*mass_length*g*theta)/(total_mass*(k_t+ mass_length) - mass_length**2)
    
    def calc_velocity_delta(self, m1, m2, l, theta_acc, F):
        return (m2*l*theta_acc + F)/(m1+m2)
    
    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        feat = self.encoder(torch.cat([obs_norm, act_norm], dim=-1))
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.head(feat)
        
        # assign raw_delta as [m1, m2, l, k_t]
        d_theta = self.calc_angular_velocity_delta(raw_delta[0], raw_delta[1], 
                                                   raw_delta[2], raw_delta[3],
                                                   obs_norm[0,:], act_norm, 9.81)
        d_velocity = self.calc_velocity_delta(raw_delta[0], raw_delta[1], raw_delta[2], d_theta, act_norm)
        delta = torch.tensor([obs_norm[1,:], obs_norm[3,:], d_velocity, d_theta])
        return delta, hidden