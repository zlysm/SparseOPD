# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def build_sparse_opd_mask(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    mode: str,
    random_fraction: float = 0.001,
    tail_fraction: float = 0.0005,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    """Select response tokens for sparse OPD, independently within each sequence."""
    if rewards.shape != response_mask.shape:
        raise ValueError(
            f"rewards and response_mask must have the same shape, got {rewards.shape} and {response_mask.shape}."
        )
    if rewards.ndim != 2:
        raise ValueError(f"Expected [batch, response_length] rewards, got shape {rewards.shape}.")

    valid = response_mask.bool()
    selected = torch.zeros_like(valid)
    if mode in {"none", "dense"}:
        return valid
    if mode == "randmask":
        return (torch.rand(rewards.shape, device=rewards.device) < random_fraction) & valid
    if mode == "at_lt":
        if threshold is None:
            raise ValueError("threshold is required for at_lt.")
        return (rewards < threshold) & valid
    if mode == "at_gt":
        if threshold is None:
            raise ValueError("threshold is required for at_gt.")
        return (rewards > threshold) & valid

    for row in range(rewards.shape[0]):
        indices = torch.nonzero(valid[row], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        row_rewards = rewards[row, indices]
        if mode == "rand1tok":
            chosen = torch.randint(indices.numel(), (1,), device=indices.device)
            selected[row, indices[chosen]] = True
        elif mode == "maxtok":
            selected[row, indices[row_rewards.argmax()]] = True
        elif mode == "mintok":
            selected[row, indices[row_rewards.argmin()]] = True
        elif mode == "minmaxtok":
            selected[row, indices[row_rewards.argmax()]] = True
            selected[row, indices[row_rewards.argmin()]] = True
        elif mode == "pctltail":
            lower = torch.quantile(row_rewards.float(), tail_fraction)
            upper = torch.quantile(row_rewards.float(), 1.0 - tail_fraction)
            selected[row, indices[(row_rewards <= lower) | (row_rewards >= upper)]] = True
        else:
            raise ValueError(f"Unsupported sparse OPD mode: {mode!r}.")
    return selected


def sparse_opd_metrics(
    rewards: torch.Tensor, response_mask: torch.Tensor, sparse_mask: torch.Tensor
) -> dict[str, Metric]:
    valid = response_mask.bool()
    selected = sparse_mask.bool() & valid
    valid_rewards = rewards[valid].float()
    selected_rewards = rewards[selected].float()
    zero = rewards.new_zeros((), dtype=torch.float32)
    selected_mean = selected_rewards.mean() if selected_rewards.numel() else zero
    selected_std = selected_rewards.std(unbiased=False) if selected_rewards.numel() else zero
    selected_min = selected_rewards.min() if selected_rewards.numel() else zero
    selected_max = selected_rewards.max() if selected_rewards.numel() else zero
    reward_min = valid_rewards.min() if valid_rewards.numel() else zero
    reward_max = valid_rewards.max() if valid_rewards.numel() else zero
    return {
        "sparse_opd/keep_fraction": Metric(AggregationType.MEAN, selected.sum().float() / valid.sum().clamp_min(1)),
        "sparse_opd/tokens_per_response": Metric(
            AggregationType.MEAN, selected.sum(dim=-1).float().mean() if selected.shape[0] else zero
        ),
        "sparse_opd/reward_mean": Metric(AggregationType.MEAN, valid_rewards.mean() if valid_rewards.numel() else zero),
        "sparse_opd/reward_std": Metric(
            AggregationType.MEAN, valid_rewards.std(unbiased=False) if valid_rewards.numel() else zero
        ),
        "sparse_opd/reward_min": Metric(AggregationType.MIN, reward_min),
        "sparse_opd/reward_max": Metric(AggregationType.MAX, reward_max),
        "sparse_opd/selected_reward_mean": Metric(AggregationType.MEAN, selected_mean),
        "sparse_opd/selected_reward_std": Metric(AggregationType.MEAN, selected_std),
        "sparse_opd/selected_reward_min": Metric(AggregationType.MIN, selected_min),
        "sparse_opd/selected_reward_max": Metric(AggregationType.MAX, selected_max),
        "sparse_opd/selected_positive_fraction": Metric(
            AggregationType.MEAN,
            (selected_rewards > 0).float().mean() if selected_rewards.numel() else zero,
        ),
    }


def paper_clipped_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    sparse_mask: torch.Tensor,
    loss_config: DistillationLossConfig,
    sparse_opd_loss_agg_mode: str,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Metric]]:
    """Compute Sparse OPD's clipped surrogate with per-response token means.

    The sparse mask gates only the numerator. Each selected-token sum is divided
    by that response's full valid length, then response losses are averaged over
    the global PPO minibatch, as in the paper's objective.
    """
    if sparse_opd_loss_agg_mode != "paper-seq-mean-token-mean":
        raise ValueError("Paper-style Sparse OPD requires sparse_opd_loss_agg_mode='paper-seq-mean-token-mean'.")
    if response_mask.shape != sparse_mask.shape:
        raise ValueError(
            f"response_mask and sparse_mask must have the same shape, "
            f"got {response_mask.shape} and {sparse_mask.shape}."
        )

    valid_response_mask = response_mask.bool()
    selected_mask = sparse_mask.bool() & valid_response_mask
    log_ratio = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(
        min=1.0 - loss_config.clip_ratio_low,
        max=1.0 + loss_config.clip_ratio_high,
    )
    pg_losses = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    response_lengths = valid_response_mask.sum(dim=-1)
    per_response_losses = (pg_losses * selected_mask).sum(dim=-1) / response_lengths.clamp_min(1)
    nonempty_responses = response_lengths > 0
    loss = agg_loss(
        loss_mat=per_response_losses.unsqueeze(-1),
        loss_mask=nonempty_responses.unsqueeze(-1),
        loss_agg_mode="seq-mean-token-sum",
        **loss_config.global_batch_info,
    )
    selected_count = selected_mask.sum().clamp_min(1)
    clip_mask = ((advantages > 0) & (ratio > 1.0 + loss_config.clip_ratio_high)) | (
        (advantages < 0) & (ratio < 1.0 - loss_config.clip_ratio_low)
    )
    metrics = {
        "distillation/pg_clipfrac": Metric(
            AggregationType.MEAN, ((clip_mask & selected_mask).sum().float() / selected_count)
        ),
        "distillation/ppo_kl": Metric(AggregationType.MEAN, ((-log_ratio) * selected_mask).sum() / selected_count),
        "distillation/pg_clipfrac_lower": Metric(AggregationType.MEAN, log_ratio.new_zeros(())),
    }
    return loss, metrics


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    paper_sparse_opd = loss_config.sparse_opd_mode != "none"
    if loss_config.loss_max_clamp is not None and not paper_sparse_opd:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if paper_sparse_opd:
        loss_config.global_batch_info.update(config.global_batch_info)
        loss_config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
        response_mask = response_mask.to_padded_tensor(False) if response_mask.is_nested else response_mask.bool()
        sparse_mask = (
            data["sparse_opd_mask"].to_padded_tensor(False)
            if data["sparse_opd_mask"].is_nested
            else data["sparse_opd_mask"].bool()
        )
        advantages = (
            data["sparse_opd_reward"].to_padded_tensor(0.0)
            if data["sparse_opd_reward"].is_nested
            else data["sparse_opd_reward"]
        )
        distillation_metrics.update(sparse_opd_metrics(advantages, response_mask, sparse_mask))
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = old_log_prob.to_padded_tensor(0.0)
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        rollout_is_weights = data.get("rollout_is_weights", None)
        if rollout_is_weights is not None and rollout_is_weights.is_nested:
            rollout_is_weights = rollout_is_weights.to_padded_tensor(0.0)
        distillation_loss, pg_metrics = paper_clipped_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages.detach(),
            response_mask=response_mask,
            sparse_mask=sparse_mask,
            loss_config=loss_config,
            sparse_opd_loss_agg_mode=loss_config.sparse_opd_loss_agg_mode,
            rollout_is_weights=rollout_is_weights,
        )
        distillation_metrics.update(pg_metrics)
    elif loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
