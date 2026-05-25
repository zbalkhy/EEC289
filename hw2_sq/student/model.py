"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
        dt: float = 0.04,
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
        self.head = nn.Linear(hidden_dim, 6)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        self.register_buffer("delta_std", torch.ones(obs_dim))
        self.register_buffer("dt", torch.tensor(float(dt), dtype=torch.float32))

    @torch.jit.ignore
    def set_normalizer(self, normalizer) -> None:
        with torch.no_grad():
            self._copy_buffer_if_changed(self.obs_mean, normalizer.obs_mean)
            self._copy_buffer_if_changed(self.obs_std, normalizer.obs_std)
            self._copy_buffer_if_changed(self.act_mean, normalizer.act_mean)
            self._copy_buffer_if_changed(self.act_std, normalizer.act_std)
            self._copy_buffer_if_changed(self.delta_mean, normalizer.delta_mean)
            self._copy_buffer_if_changed(self.delta_std, normalizer.delta_std)

    @torch.jit.ignore
    def _copy_buffer_if_changed(self, buffer: torch.Tensor, value) -> None:
        value_t = torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device)
        if not torch.equal(buffer, value_t):
            buffer.copy_(value_t)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def calc_angular_velocity_delta(
        self,
        m1,
        m2,
        l,
        k_t,
        b,
        c,
        theta,
        x_dot,
        theta_dot,
        force,
        g,
    ):
        total_mass = m1 + m2
        mass_length = m2 * l
        pole_inertia = k_t + mass_length * l
        cart_drive = force - b * x_dot
        pole_drive = mass_length * g * theta - c * theta_dot
        numerator = total_mass * pole_drive - mass_length * cart_drive
        denominator = total_mass * pole_inertia - mass_length**2 + 1e-6
        return numerator / denominator

    def calc_velocity_delta(self, m1, m2, l, theta_acc, force, x_dot, b):
        return (force - b * x_dot - m2 * l * theta_acc) / (m1 + m2 + 1e-6)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        feat = self.encoder(torch.cat([obs_norm, act_norm], dim=-1))
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.head(feat)

        obs = obs_norm * self.obs_std + self.obs_mean
        act = act_norm * self.act_std + self.act_mean

        # Interpret the network head as positive physical parameters:
        # cart mass, pole mass, pole length, pole inertia, cart friction,
        # and pendulum hinge damping.
        params = F.softplus(raw_delta) + 1e-4
        m1 = params[:, 0]
        m2 = params[:, 1]
        l = params[:, 2]
        k_t = params[:, 3]
        b = params[:, 4]
        c = params[:, 5]

        theta = obs[:, 1]
        x_dot = obs[:, 2]
        theta_dot = obs[:, 3]
        force = 100 * act[:, 0]

        d_theta = self.calc_angular_velocity_delta(
            m1,
            m2,
            l,
            k_t,
            b,
            c,
            theta,
            x_dot,
            theta_dot,
            force,
            9.81,
        )
        d_velocity = self.calc_velocity_delta(m1, m2, l, d_theta, force, x_dot, b)
        delta_phys = torch.stack(
            [x_dot * self.dt, theta_dot * self.dt, d_velocity * self.dt, d_theta * self.dt],
            dim=-1,
        )
        delta = (delta_phys - self.delta_mean) / self.delta_std
        delta = self.delta_limit * torch.tanh(delta / self.delta_limit)

        return delta, hidden
