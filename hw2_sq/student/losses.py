"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout

_LOSS_CALLS = 0
_LAST_MODEL_ID = None


def _next_loss_call(model) -> int:
    global _LOSS_CALLS, _LAST_MODEL_ID

    model_id = id(model)
    if _LAST_MODEL_ID != model_id:
        _LAST_MODEL_ID = model_id
        _LOSS_CALLS = 0
    _LOSS_CALLS += 1
    return _LOSS_CALLS


def rollout_curriculum_horizon(loss_cfg: dict, update: int) -> int:
    curriculum = loss_cfg.get("rollout_curriculum", {})
    if not curriculum.get("enabled", False):
        return int(loss_cfg.get("rollout_train_horizon", 5))

    update = max(int(update), 1)
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
    if hidden is None:
        obs = states[:, :-1].reshape(-1, states.shape[-1])
        act = actions.reshape(-1, actions.shape[-1])
        target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
        obs_norm = normalizer.normalize_obs(obs)
        act_norm = normalizer.normalize_act(act)
        target_norm = normalizer.normalize_delta(target_delta)
        pred_norm, _ = model(obs_norm, act_norm, None)
        return F.mse_loss(pred_norm, target_norm)

    step_losses = []
    for t in range(actions.shape[1]):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        target_delta = states[:, t + 1] - states[:, t]
        target_norm = normalizer.normalize_delta(target_delta)

        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        step_losses.append(F.mse_loss(pred_norm, target_norm, reduction="none"))

    return torch.stack(step_losses, dim=1).mean()


def denoising_one_step_delta_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    noise_std: float,
) -> torch.Tensor:
    if states.shape[1] != actions.shape[1] + 1:
        raise ValueError(
            "denoising one-step loss expects states to have exactly one more time step than actions: "
            f"got states={states.shape[1]}, actions={actions.shape[1]}."
        )
    if float(noise_std) <= 0.0:
        return one_step_delta_loss(model, states, actions, normalizer)

    obs_std = torch.as_tensor(normalizer.obs_std, dtype=states.dtype, device=states.device)
    hidden = model.initial_hidden(states.shape[0], states.device)
    if hidden is None:
        obs = states[:, :-1].reshape(-1, states.shape[-1])
        act = actions.reshape(-1, actions.shape[-1])
        target_next = states[:, 1:].reshape(-1, states.shape[-1])
        noise_norm = torch.randn_like(obs) * float(noise_std)
        noisy_obs = obs + noise_norm * obs_std
        noisy_obs_norm = normalizer.normalize_obs(obs) + noise_norm
        act_norm = normalizer.normalize_act(act)
        target_norm = normalizer.normalize_delta(target_next - noisy_obs)
        pred_norm, _ = model(noisy_obs_norm, act_norm, None)
        return F.mse_loss(pred_norm, target_norm)

    step_losses = []
    for t in range(actions.shape[1]):
        obs = states[:, t]
        noise_norm = torch.randn_like(obs) * float(noise_std)
        noisy_obs = obs + noise_norm * obs_std
        noisy_obs_norm = normalizer.normalize_obs(obs) + noise_norm
        act_norm = normalizer.normalize_act(actions[:, t])
        target_norm = normalizer.normalize_delta(states[:, t + 1] - noisy_obs)

        pred_norm, hidden = model(noisy_obs_norm, act_norm, hidden)
        step_losses.append(F.mse_loss(pred_norm, target_norm, reduction="none"))

    return torch.stack(step_losses, dim=1).mean()


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    *,
    windows_per_batch: int = 1,
    cap_nmse: float | None = 1.0,
    margin_threshold: float | None = None,
    margin_weight: float = 0.0,
) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    losses = []
    for _ in range(max(int(windows_per_batch), 1)):
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
        per_step_nmse = torch.mean((pred_norm - target_norm) ** 2, dim=-1)
        # this is for long rollout
        if cap_nmse is not None and float(cap_nmse) > 0.0:
            base = float(cap_nmse) * torch.log1p(per_step_nmse / float(cap_nmse)).mean()
        else:
            base = per_step_nmse.mean()
        roll = base
        # this is for short rollout
        if margin_threshold is not None and float(margin_weight) > 0.0:
            threshold_loss = F.softplus(20.0 * (per_step_nmse - float(margin_threshold))).mean() / 20.0
            if cap_nmse is not None and float(cap_nmse) > 0.0:
                threshold_loss = torch.clamp(threshold_loss, max=float(cap_nmse))
            roll = roll + float(margin_weight) * threshold_loss
        losses.append(roll)
    return torch.stack(losses).mean()


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    if float(loss_cfg.get("denoise_one_step_weight", 0.0)) > 0.0:
        denoise = denoising_one_step_delta_loss(
            model,
            states,
            actions,
            normalizer,
            noise_std=float(loss_cfg.get("denoise_obs_noise_std", 0.02)),
        )
    else:
        denoise = states.new_tensor(0.0)
    update = _next_loss_call(model)
    long_horizon = rollout_curriculum_horizon(loss_cfg, update)
    short_horizon = int(loss_cfg.get("short_rollout_horizon", 0))

    warmup = int(cfg["eval"].get("warmup_steps", 5))
    if short_horizon > 0 and float(loss_cfg.get("short_rollout_weight", 0.0)) > 0.0:
        short_roll = rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps=warmup,
            horizon=short_horizon,
            windows_per_batch=int(loss_cfg.get("short_rollout_windows_per_batch", 1)),
            cap_nmse=loss_cfg.get("short_rollout_cap_nmse", None),
            margin_threshold=loss_cfg.get("short_rollout_margin_threshold"),
            margin_weight=float(loss_cfg.get("short_rollout_margin_weight", 0.0)),
        )
    else:
        short_roll = states.new_tensor(0.0)

    long_roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=long_horizon,
        windows_per_batch=int(loss_cfg.get("rollout_windows_per_batch", 1)),
        cap_nmse=loss_cfg.get("long_rollout_cap_nmse", 1.0),
        margin_threshold=loss_cfg.get("long_rollout_margin_threshold"),
        margin_weight=float(loss_cfg.get("long_rollout_margin_weight", 0.0)),
    )
    total = (
        float(loss_cfg.get("one_step_weight", 1.0)) * one
        + float(loss_cfg.get("denoise_one_step_weight", 0.0)) * denoise
        + float(loss_cfg.get("short_rollout_weight", 0.0)) * short_roll
        + float(loss_cfg.get("rollout_weight", 0.3)) * long_roll
    )
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/denoise_one_step": float(denoise.detach().cpu()),
        "loss/short_rollout": float(short_roll.detach().cpu()),
        "loss/short_rollout_horizon": float(short_horizon),
        "loss/rollout": float(long_roll.detach().cpu()),
        "loss/rollout_horizon": float(long_horizon),
    }
