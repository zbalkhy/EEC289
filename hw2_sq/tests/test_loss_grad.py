from __future__ import annotations

import numpy as np
import torch

from student.losses import compute_loss, one_step_delta_loss
from student.model import StudentWorldModel
from wm_hw.normalizer import Normalizer


def test_loss_backward_has_gradients():
    states = torch.randn(4, 12, 4)
    actions = torch.randn(4, 11, 1)
    norm = Normalizer.from_train(states.numpy(), actions.numpy())
    model = StudentWorldModel(hidden_dim=32)
    cfg = {"loss": {"one_step_weight": 1.0, "rollout_weight": 0.3, "rollout_train_horizon": 5}, "eval": {"warmup_steps": 5}}
    loss, metrics = compute_loss(model, {"states": states, "actions": actions}, norm, cfg)
    loss.backward()
    assert "loss/rollout" in metrics
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def test_one_step_loss_carries_hidden_through_teacher_forced_sequence():
    class CountingModel:
        def __init__(self):
            self.hidden_inputs = []

        def initial_hidden(self, batch_size, device):
            return torch.zeros(batch_size, 1, device=device)

        def __call__(self, obs_norm, act_norm, hidden):
            self.hidden_inputs.append(hidden.detach().clone())
            return torch.zeros_like(obs_norm), hidden + 1.0

    states = torch.zeros(2, 4, 1)
    actions = torch.zeros(2, 3, 1)
    norm = Normalizer(
        obs_mean=np.zeros(1, dtype=np.float32),
        obs_std=np.ones(1, dtype=np.float32),
        act_mean=np.zeros(1, dtype=np.float32),
        act_std=np.ones(1, dtype=np.float32),
        delta_mean=np.zeros(1, dtype=np.float32),
        delta_std=np.ones(1, dtype=np.float32),
    )
    model = CountingModel()

    loss = one_step_delta_loss(model, states, actions, norm)

    assert torch.isclose(loss, torch.tensor(0.0))
    assert len(model.hidden_inputs) == actions.shape[1]
    for t, hidden in enumerate(model.hidden_inputs):
        assert torch.equal(hidden, torch.full_like(hidden, float(t)))
