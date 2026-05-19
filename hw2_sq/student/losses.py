"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout

_LOSS_CALLS = 0


def rollout_curriculum_horizon(loss_cfg: dict) -> int:
    curriculum = loss_cfg.get("rollout_curriculum", {})
    if not curriculum.get("enabled", False):
        return int(loss_cfg.get("rollout_train_horizon", 5))

    update = max(_LOSS_CALLS, 1)
    start = int(curriculum.get("start_horizon", 5))
    end = int(curriculum.get("end_horizon", loss_cfg.get("rollout_train_horizon", start)))
    warmup = int(curriculum.get("warmup_updates", 0))
    ramp = max(int(curriculum.get("ramp_updates", 1)), 1)

    if update <= warmup:
        return start

    progress = min((update - warmup) / ramp, 1.0)
    horizon = round(start + progress * (end - start))
    return max(1, int(horizon))


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    if states.shape[1] != actions.shape[1] + 1:
        raise ValueError(
            "one-step loss expects states to have exactly one more time step than actions: "
            f"got states={states.shape[1]}, actions={actions.shape[1]}."
        )

    hidden = model.initial_hidden(states.shape[0], states.device)
    step_losses = []
    for t in range(actions.shape[1]):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        target_delta = states[:, t + 1] - states[:, t]
        target_norm = normalizer.normalize_delta(target_delta)

        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        step_losses.append(F.mse_loss(pred_norm, target_norm, reduction="none"))

    return torch.stack(step_losses, dim=1).mean()


def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    weights = torch.sqrt(torch.arange(
        1, target_norm.shape[1]+1,
        device=target_norm.device,
        dtype=target_norm.dtype
    )).view(1,-1,1).expand_as(target_norm)
    weights = weights / weights.mean()
    return F.mse_loss(pred_norm, target_norm, weight=weights)

    # window_losses = []
    # for i in range(0, max_start, max(max_start // 3, 1)):
    #     start = i
    #     sub_states = states[:, start : start + needed_states]
    #     sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    #     preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    #     targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    #     pred_norm = normalizer.normalize_obs(preds)
    #     target_norm = normalizer.normalize_obs(targets)
    #     window_losses.append((i/max_start)*F.mse_loss(pred_norm, target_norm))
    # return sum(window_losses)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):

    global _LOSS_CALLS
    _LOSS_CALLS += 1


    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = rollout_curriculum_horizon(loss_cfg)

    warmup = int(cfg["eval"].get("warmup_steps", 5))
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
    }
