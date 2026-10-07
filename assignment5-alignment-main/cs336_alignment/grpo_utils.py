import torch
from typing import Any, Callable, Literal
from transformers import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange, reduce
from torch.optim import Optimizer


# 把问题和答案拼接为一条数据 token id序列 （带mask的）
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:

    assert len(prompt_strs) == len(output_strs)

    prompt_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    output_ids = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    combined_ids = []
    response_mask = []
    for a, b in zip(prompt_ids, output_ids):
        combined_ids.append(torch.tensor(a + b, dtype=torch.long))
        response_mask.append(
            torch.tensor([False] * len(a) + [True] * len(b), dtype=torch.bool)
        )

    combined_ids = pad_sequence(
        combined_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    response_mask = pad_sequence(response_mask, batch_first=True, padding_value=False)

    return {
        "input_ids": combined_ids[:, :-1],
        "labels": combined_ids[:, 1:],
        "response_mask": response_mask[:, 1:],
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits  # B S V
    # softmax
    log_probs = torch.log_softmax(logits, dim=-1)
    label_indices = rearrange(labels, "B S -> B S 1")

    # 取出label对应概率
    selected_log_probs = torch.gather(log_probs, dim=-1, index=label_indices)
    selected_log_probs = rearrange(selected_log_probs, "B S 1 -> B S")
    token_entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)

    if return_token_entropy:
        return {"log_probs": selected_log_probs, "token_entropy": token_entropy}
    return {"log_probs": selected_log_probs}


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:

    assert len(rollout_responses) == len(repeated_ground_truths)

    reward_records = [
        reward_fn(a, b) for a, b in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [dic["reward"] for dic in reward_records], dtype=torch.float32
    )

    metadata = {
        "mean_reward": sum(dic["reward"] for dic in reward_records)
        / len(reward_records),
        "mean_format_reward": sum(dic["format_reward"] for dic in reward_records)
        / len(reward_records),
    }

    return (raw_rewards, metadata)


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):

    rewards_grouped = rearrange(raw_rewards, "(B g) -> B g", g=group_size)
    mean_rewards = reduce(rewards_grouped, "B g -> B 1", "mean")
    std_rewards = rewards_grouped.std(keepdim=True, dim=-1)
    advantages = rewards_grouped
    if baseline == "mean":
        advantages = advantages - mean_rewards
    if advantage_normalizer == "std":
        advantages = advantages / (std_rewards + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (mean_rewards + advantage_eps)

    advantages = rearrange(advantages, "n g -> (n g)", g=group_size)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_group_std": std_rewards.mean().item(),
        "zero_variance_group_fraction": (std_rewards == 0).float().mean().item(),
    }

    return (advantages, metadata)


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # [BG,]
    policy_log_probs: torch.Tensor,  # [BG, S]
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(raw_rewards_or_advantages.shape) == 1:
        raw_rewards_or_advantages = rearrange(raw_rewards_or_advantages, "M -> M 1")
    per_token_policy_gradient_loss = -policy_log_probs * raw_rewards_or_advantages
    metadata = {}

    return (per_token_policy_gradient_loss, metadata)


# 对于M = B*G 个rollout 的loss求和 得到总的g
def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_per_token_loss = mask * per_token_policy_gradient_loss

    if loss_normalization == "sequence":
        # [M]：每条回答的有效 token 数，避免空回答除以0
        lengths = mask.sum(dim=-1).clamp_min(1)

        # [M]：非空回答正常平均；空回答贡献0
        sequence_losses = masked_per_token_loss.sum(dim=-1) / lengths

        # 标量：对整个 microbatch 平均
        loss = sequence_losses.mean()

    elif loss_normalization == "constant":
        if normalization_constant is None or normalization_constant <= 0:
            raise ValueError("normalization_constant must be positive")
        loss = masked_per_token_loss.sum() / normalization_constant
    else:
        raise NotImplementedError

    return loss


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    # 当前只实现标准 on-policy GRPO
    if importance_reweighting_method != "none":
        raise NotImplementedError
    if loss_normalization != "sequence":
        raise NotImplementedError
    batch_size = len(rollout_responses)
    assert batch_size % gradient_accumulation_steps == 0
    assert gradient_accumulation_steps > 0
    assert len(repeated_prompts) == len(repeated_ground_truths) == batch_size

    tokenized = tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer
    )

    lengths = tokenized["response_mask"].sum(dim=-1)
    empty_indices = (lengths == 0).nonzero(as_tuple=True)[0].tolist()

    if empty_indices:
        print("Zero-token response indices:", empty_indices, flush=True)
        for j in empty_indices[:5]:
            print("Response:", repr(rollout_responses[j]), flush=True)


    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    device = next(model.parameters()).device

    advantages = advantages.to(device)
    inputs = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    metadata = {**reward_metadata, **advantage_metadata}
    total_loss = torch.zeros((), device=device)
    microbatch_size = len(inputs) // gradient_accumulation_steps

    model.train()
    optimizer.zero_grad()

    entropy_sum = torch.zeros((), device=device)
    response_token_count = torch.zeros((), device = device)

    for i in range(0, len(inputs), microbatch_size):
        inputs_microbatch = inputs[i : i + microbatch_size]
        labels_microbatch = labels[i : i + microbatch_size]
        response_mask_microbatch = response_mask[i : i + microbatch_size]
        log_probs_dict = get_response_log_probs(
            model, inputs_microbatch, labels_microbatch, return_token_entropy=True
        )
        policy_log_probs = log_probs_dict["log_probs"]
        with torch.no_grad():
            entropy = log_probs_dict['token_entropy'].detach()
            entropy_sum += (
                entropy.float() * response_mask_microbatch
            ).sum()
            response_token_count += response_mask_microbatch.sum()

        per_token_policy_gradient_loss, _ = compute_policy_gradient_loss(
            advantages[i : i + microbatch_size],
            policy_log_probs,
            importance_reweighting_method,
            old_log_probs,
            cliprange,
            response_mask_microbatch,
        )

        loss = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss,
                response_mask_microbatch,
                loss_normalization,
                normalization_constant,
            )
            / gradient_accumulation_steps
        )
        total_loss += loss.detach()
        # Backward pass.
        loss.backward()
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        metadata["grad_norm"] = grad_norm.detach().item()
    # Update weights once across entire batch.
    optimizer.step()
    # Zero gradients once across entire batch.
    optimizer.zero_grad(set_to_none = True)
    metadata["token_entropy"] = (
        entropy_sum / response_token_count.clamp_min(1)
    ).item()

    return total_loss, metadata
