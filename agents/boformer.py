"""BOFormer agent for non-Markovian multi-objective Bayesian optimization.

The implementation follows the Q-augmented representation in BOFormer
(ICLR 2025). A selected point at timestep ``t`` is represented by its GP
posterior statistics, incumbent objective values, and normalized budget. Each
historical timestep contributes three causal tokens: observation-action,
target-network Q-value, and reward. The three tokens share the same learned
GPT-2 positional embedding.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_gpt2(
    *,
    hidden_dim: int,
    n_layers: int,
    n_heads: int,
    window_size: int,
    dropout: float,
):
    try:
        from transformers import GPT2Config, GPT2Model
    except ImportError as exc:
        raise ImportError(
            "BOFormer requires Hugging Face transformers. "
            "Install dependencies from requirements.txt."
        ) from exc

    max_tokens = int(window_size)
    config = GPT2Config(
        vocab_size=1,
        bos_token_id=None,
        eos_token_id=None,
        n_embd=int(hidden_dim),
        n_layer=int(n_layers),
        n_head=int(n_heads),
        n_positions=max_tokens,
        n_ctx=max_tokens,
        resid_pdrop=float(dropout),
        embd_pdrop=float(dropout),
        attn_pdrop=float(dropout),
        use_cache=False,
    )
    return GPT2Model(config)


def upgrade_legacy_observation_action_weight(
    state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    """Convert legacy ``[hidden, m, 4]`` BOFormer embeddings to ``[hidden, 3m+1]``.

    Older local checkpoints repeated ``t/T`` once per objective. The new
    parameterization matches upstream by keeping a single shared progress
    feature, so the migrated progress column sums the legacy per-objective
    progress weights.
    """
    if state_dict is None:
        return state_dict
    obs_weight = state_dict.get("observation_action_weight")
    if obs_weight is None or int(obs_weight.ndim) != 3:
        return state_dict
    migrated = dict(state_dict)
    migrated["observation_action_weight"] = torch.cat(
        [
            obs_weight[:, :, 0],
            obs_weight[:, :, 1],
            obs_weight[:, :, 2],
            obs_weight[:, :, 3].sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    return migrated


class BOFormer(nn.Module):
    """GPT-2 Q-function with BOFormer's Q-augmented trajectory encoding.

    Parameters follow the paper defaults. ``n_objectives`` is fixed for one
    model because the observation-action embedding has size ``3 * m + 1``. The
    transformer weights can still be transferred between models with different
    objective counts via :meth:`load_transformer_weights_from`.

    ``window_size`` is interpreted as the total Transformer context length in
    tokens. Since each historical timestep contributes three tokens
    (observation-action, Q-value, reward) and the current candidate adds one
    token, the effective number of retained history steps is
    ``floor((window_size - 1) / 3)``.

    All objective statistics should be normalized consistently before being
    passed to the model. The class is agnostic to minimization/maximization; the
    caller supplies the corresponding incumbent ``best_objectives`` values.
    """

    uses_q_augmented_history = True

    def __init__(
        self,
        n_objectives: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 4,
        window_size: int = 31,
        dropout: float = 0.1,
        temperature: float = 1000.0,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_objectives = int(n_objectives)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.window_size = int(window_size)
        self.dropout = float(dropout)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)

        if self.n_objectives <= 0:
            raise ValueError(f"n_objectives must be positive, got {n_objectives}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        if self.n_heads <= 0 or self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}."
            )
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        self.observation_action_weight = nn.Parameter(
            torch.empty(self.hidden_dim, 3 * self.n_objectives + 1)
        )
        self.observation_action_bias = nn.Parameter(torch.zeros(self.hidden_dim))
        self.reward_embedding = nn.Linear(1, self.hidden_dim)
        self.q_value_embedding = nn.Linear(1, self.hidden_dim)
        self.transformer = _build_gpt2(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            window_size=self.window_size,
            dropout=self.dropout,
        )
        self.q_head = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_uniform_(self.observation_action_weight)

    @property
    def max_history_steps(self) -> int:
        return max(0, (int(self.window_size) - 1) // 3)

    @staticmethod
    def _as_float_tensor(value: Any, *, device, dtype) -> torch.Tensor:
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _prepare_candidates(
        self,
        candidate_mean,
        candidate_variance,
        best_objectives,
        progress,
        objective_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = next(self.parameters())
        device, dtype = reference.device, reference.dtype
        mean = self._as_float_tensor(candidate_mean, device=device, dtype=dtype)
        variance = self._as_float_tensor(candidate_variance, device=device, dtype=dtype)
        if mean.dim() == 2:
            mean = mean.unsqueeze(0)
        if variance.dim() == 2:
            variance = variance.unsqueeze(0)
        if mean.dim() != 3 or mean.size(-1) != self.n_objectives:
            raise ValueError(
                "candidate_mean must have shape [B, N, m] or [N, m], "
                f"got {tuple(mean.shape)}."
            )
        if tuple(variance.shape) != tuple(mean.shape):
            raise ValueError(
                f"candidate_variance shape {tuple(variance.shape)} must match {tuple(mean.shape)}."
            )

        batch_size = int(mean.size(0))
        best = self._as_float_tensor(best_objectives, device=device, dtype=dtype)
        if best.dim() == 1:
            best = best.unsqueeze(0)
        if best.size(0) == 1 and batch_size > 1:
            best = best.expand(batch_size, -1)
        if tuple(best.shape) != (batch_size, self.n_objectives):
            raise ValueError(
                f"best_objectives must have shape {(batch_size, self.n_objectives)}, "
                f"got {tuple(best.shape)}."
            )

        step = self._as_float_tensor(progress, device=device, dtype=dtype)
        if step.dim() == 0:
            step = step.repeat(batch_size)
        step = step.reshape(batch_size, -1)[:, :1].clamp(0.0, 1.0)
        if objective_mask is None:
            mask = torch.ones(batch_size, self.n_objectives, device=device, dtype=dtype)
        else:
            mask = self._as_float_tensor(objective_mask, device=device, dtype=dtype)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.size(0) == 1 and batch_size > 1:
                mask = mask.expand(batch_size, -1)
            if tuple(mask.shape) != (batch_size, self.n_objectives):
                raise ValueError(
                    f"objective_mask must have shape {(batch_size, self.n_objectives)}, "
                    f"got {tuple(mask.shape)}."
                )
            mask = mask.clamp(0.0, 1.0)
        return mean, variance.clamp_min(0.0), best, step, mask

    def build_observation_action_features(
        self,
        *,
        mean: torch.Tensor,
        variance: torch.Tensor,
        best: torch.Tensor,
        progress: torch.Tensor,
        objective_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the padded/masked state-action vector ``[mu, var, y*, t/T]``."""
        if mean.dim() not in {3, 4}:
            raise ValueError(f"mean must be rank 3 or 4, got rank {mean.dim()}.")
        mask = objective_mask.to(device=mean.device, dtype=mean.dtype)
        if mean.dim() == 3:
            _, n_items, _ = mean.shape
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.dim() != 2:
                raise ValueError(
                    f"objective_mask must have rank 1 or 2 for rank-3 mean, got rank {mask.dim()}."
                )
            best_expanded = best.unsqueeze(1).expand(-1, n_items, -1)
            progress_expanded = progress.unsqueeze(1).expand(-1, n_items, -1)
            mask_expanded = mask.unsqueeze(1).expand(-1, n_items, -1)
        else:
            _, n_steps, n_items, _ = mean.shape
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if mask.dim() != 3:
                raise ValueError(
                    f"objective_mask must have rank 2 or 3 for rank-4 mean, got rank {mask.dim()}."
                )
            best_expanded = best.unsqueeze(2).expand(-1, n_steps, n_items, -1)
            progress_expanded = progress.unsqueeze(2).expand(-1, n_steps, n_items, -1)
            mask_expanded = mask.unsqueeze(2).expand(-1, n_steps, n_items, -1)
        masked_mean = mean * mask_expanded
        masked_variance = variance * mask_expanded
        masked_best = best_expanded * mask_expanded
        return torch.cat(
            [
                masked_mean,
                masked_variance,
                masked_best,
                progress_expanded,
            ],
            dim=-1,
        )

    def _embed_observation_action_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.size(-1) != 3 * self.n_objectives + 1:
            raise ValueError(
                "features must have trailing shape "
                f"({3 * self.n_objectives + 1},), got {tuple(features.shape)}."
            )
        return F.linear(features, self.observation_action_weight, self.observation_action_bias)

    def encode_candidates(
        self,
        candidate_mean,
        candidate_variance,
        best_objectives,
        progress,
        objective_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, variance, best, step, mask = self._prepare_candidates(
            candidate_mean,
            candidate_variance,
            best_objectives,
            progress,
            objective_mask=objective_mask,
        )
        features = self.build_observation_action_features(
            mean=mean,
            variance=variance,
            best=best,
            progress=step,
            objective_mask=mask,
        )
        return features, self._embed_observation_action_features(features)

    @staticmethod
    def _prepare_history_tensor(value, *, name, batch_size, n_steps, trailing_size, device, dtype):
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        expected_rank = 3 if trailing_size > 1 else 2
        if tensor.dim() == expected_rank - 1:
            tensor = tensor.unsqueeze(0)
        if trailing_size == 1 and tensor.dim() == 3 and tensor.size(-1) == 1:
            tensor = tensor.squeeze(-1)
        if tensor.size(0) == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, *tensor.shape[1:])
        expected = (
            (batch_size, n_steps, trailing_size)
            if trailing_size > 1
            else (batch_size, n_steps)
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}.")
        return tensor

    def _prepare_history(
        self,
        *,
        history_mean,
        history_variance,
        history_best_objectives,
        history_progress,
        history_rewards,
        history_q_values,
        history_objective_mask,
        history_mask,
        batch_size,
        device,
        dtype,
    ) -> dict[str, torch.Tensor]:
        if history_mean is None:
            return {
                "observation_tokens": torch.empty(
                    batch_size, 0, self.hidden_dim, device=device, dtype=dtype
                ),
                "reward_tokens": torch.empty(
                    batch_size, 0, self.hidden_dim, device=device, dtype=dtype
                ),
                "q_tokens": torch.empty(
                    batch_size, 0, self.hidden_dim, device=device, dtype=dtype
                ),
                "mask": torch.empty(batch_size, 0, device=device, dtype=torch.bool),
            }

        required_history = {
            "history_variance": history_variance,
            "history_best_objectives": history_best_objectives,
            "history_progress": history_progress,
            "history_rewards": history_rewards,
            "history_q_values": history_q_values,
            "history_objective_mask": history_objective_mask,
        }
        missing = [name for name, value in required_history.items() if value is None]
        if missing:
            raise ValueError(
                "Q-augmented history requires all history fields; missing "
                + ", ".join(missing)
                + "."
            )

        mean = torch.as_tensor(history_mean, device=device, dtype=dtype)
        if mean.dim() == 2:
            mean = mean.unsqueeze(0)
        if mean.dim() != 3 or mean.size(-1) != self.n_objectives:
            raise ValueError(
                "history_mean must have shape [B, T, m] or [T, m], "
                f"got {tuple(mean.shape)}."
            )
        if mean.size(0) == 1 and batch_size > 1:
            mean = mean.expand(batch_size, -1, -1)
        if mean.size(0) != batch_size:
            raise ValueError(f"history batch mismatch: {mean.size(0)} vs {batch_size}.")
        n_steps = int(mean.size(1))

        variance = self._prepare_history_tensor(
            history_variance,
            name="history_variance",
            batch_size=batch_size,
            n_steps=n_steps,
            trailing_size=self.n_objectives,
            device=device,
            dtype=dtype,
        ).clamp_min(0.0)
        best = self._prepare_history_tensor(
            history_best_objectives,
            name="history_best_objectives",
            batch_size=batch_size,
            n_steps=n_steps,
            trailing_size=self.n_objectives,
            device=device,
            dtype=dtype,
        )
        progress = self._prepare_history_tensor(
            history_progress,
            name="history_progress",
            batch_size=batch_size,
            n_steps=n_steps,
            trailing_size=1,
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)
        rewards = self._prepare_history_tensor(
            history_rewards,
            name="history_rewards",
            batch_size=batch_size,
            n_steps=n_steps,
            trailing_size=1,
            device=device,
            dtype=dtype,
        )
        q_values = self._prepare_history_tensor(
            history_q_values,
            name="history_q_values",
            batch_size=batch_size,
            n_steps=n_steps,
            trailing_size=1,
            device=device,
            dtype=dtype,
        )
        objective_mask = self._prepare_history_tensor(
            history_objective_mask,
            name="history_objective_mask",
            batch_size=batch_size,
            n_steps=n_steps,
            trailing_size=self.n_objectives,
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)
        if history_mask is None:
            mask = torch.ones(batch_size, n_steps, device=device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(history_mask, device=device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.size(0) == 1 and batch_size > 1:
                mask = mask.expand(batch_size, -1)
            if tuple(mask.shape) != (batch_size, n_steps):
                raise ValueError(
                    f"history_mask must have shape {(batch_size, n_steps)}, "
                    f"got {tuple(mask.shape)}."
                )

        features = self.build_observation_action_features(
            mean=mean.unsqueeze(2),
            variance=variance.unsqueeze(2),
            best=best,
            progress=progress.unsqueeze(-1),
            objective_mask=objective_mask,
        ).squeeze(2)
        return {
            "observation_tokens": self._embed_observation_action_features(features),
            "reward_tokens": self.reward_embedding(rewards.unsqueeze(-1)),
            "q_tokens": self.q_value_embedding(q_values.unsqueeze(-1)),
            "mask": mask,
        }

    def _score_candidates(
        self,
        *,
        candidate_tokens: torch.Tensor,
        history: dict[str, torch.Tensor],
        candidate_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, n_candidates, _ = candidate_tokens.shape
        history_mask = history["mask"]
        if self.max_history_steps > 0:
            valid_indices = [
                torch.nonzero(history_mask[row], as_tuple=False).squeeze(-1)[-self.max_history_steps :]
                for row in range(batch_size)
            ]
        else:
            valid_indices = [
                torch.zeros(0, dtype=torch.long, device=candidate_tokens.device)
                for _ in range(batch_size)
            ]
        max_history = max((int(indices.numel()) for indices in valid_indices), default=0)
        max_tokens = 3 * max_history + 1
        rows_total = batch_size * n_candidates
        sequence = torch.zeros(
            rows_total,
            max_tokens,
            self.hidden_dim,
            device=candidate_tokens.device,
            dtype=candidate_tokens.dtype,
        )
        attention_mask = torch.zeros(
            rows_total,
            max_tokens,
            device=candidate_tokens.device,
            dtype=torch.long,
        )
        position_ids = torch.zeros_like(attention_mask)
        current_positions = torch.empty(rows_total, device=candidate_tokens.device, dtype=torch.long)

        for batch_idx in range(batch_size):
            valid = valid_indices[batch_idx]
            n_valid = int(valid.numel())
            row_slice = slice(batch_idx * n_candidates, (batch_idx + 1) * n_candidates)
            if n_valid > 0:
                step_tokens = torch.stack(
                    [
                        history["observation_tokens"][batch_idx, valid],
                        history["q_tokens"][batch_idx, valid],
                        history["reward_tokens"][batch_idx, valid],
                    ],
                    dim=1,
                ).reshape(3 * n_valid, self.hidden_dim)
                sequence[row_slice, : 3 * n_valid] = step_tokens.unsqueeze(0).expand(
                    n_candidates, -1, -1
                )
                attention_mask[row_slice, : 3 * n_valid] = 1
                shared_positions = torch.arange(
                    n_valid, device=candidate_tokens.device, dtype=torch.long
                ).repeat_interleave(3)
                position_ids[row_slice, : 3 * n_valid] = shared_positions

            current_position = 3 * n_valid
            sequence[row_slice, current_position] = candidate_tokens[batch_idx]
            attention_mask[row_slice, current_position] = 1
            position_ids[row_slice, current_position] = n_valid
            current_positions[row_slice] = current_position

        outputs = self.transformer(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        row_indices = torch.arange(rows_total, device=candidate_tokens.device)
        current_hidden = outputs.last_hidden_state[row_indices, current_positions]
        q_values = self.q_head(current_hidden).reshape(batch_size, n_candidates)
        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask, -1e9)
        return q_values

    @staticmethod
    def _prepare_candidate_mask(candidate_mask, *, batch_size, n_candidates, device):
        if candidate_mask is None:
            return torch.ones(batch_size, n_candidates, device=device, dtype=torch.bool)
        mask = torch.as_tensor(candidate_mask, device=device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.size(0) == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, n_candidates):
            raise ValueError(
                f"candidate_mask must have shape {(batch_size, n_candidates)}, "
                f"got {tuple(mask.shape)}."
            )
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every batch item must contain at least one valid candidate.")
        return mask

    def _decode(self, q_values, candidate_mask, decode_type, epsilon, temperature):
        masked_q = q_values.masked_fill(~candidate_mask, -1e9)
        policy_probs = torch.softmax(masked_q * float(temperature), dim=-1)
        greedy = torch.argsort(masked_q, dim=-1, descending=True)
        if decode_type in {"greedy", "q_greedy"}:
            return greedy, policy_probs
        if decode_type == "softmax_sample":
            sampled = torch.multinomial(policy_probs, num_samples=1)
            ranking = []
            for row_idx in range(q_values.size(0)):
                remainder = greedy[row_idx][greedy[row_idx] != sampled[row_idx, 0]]
                ranking.append(torch.cat([sampled[row_idx], remainder], dim=0))
            return torch.stack(ranking, dim=0), policy_probs
        if decode_type == "epsilon_greedy":
            ranking = greedy.clone()
            for row_idx in range(q_values.size(0)):
                if torch.rand((), device=q_values.device) < float(epsilon):
                    valid = torch.nonzero(candidate_mask[row_idx], as_tuple=False).squeeze(-1)
                    chosen = valid[torch.randint(valid.numel(), (1,), device=q_values.device)]
                    remainder = greedy[row_idx][greedy[row_idx] != chosen]
                    ranking[row_idx] = torch.cat([chosen, remainder], dim=0)
            return ranking, policy_probs
        raise ValueError(f"Unknown decode_type: {decode_type}")

    def forward(
        self,
        candidate_mean,
        candidate_variance,
        best_objectives,
        progress,
        *,
        history_mean=None,
        history_variance=None,
        history_best_objectives=None,
        history_progress=None,
        history_rewards=None,
        history_q_values=None,
        objective_mask=None,
        history_objective_mask=None,
        history_mask=None,
        candidate_mask=None,
        decode_type: str = "softmax_sample",
        epsilon: float | None = None,
        temperature: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score all current candidates under a Q-augmented history.

        ``history_q_values`` should be generated recursively by a frozen target
        BOFormer, as specified by Algorithm 1 of the paper.
        """
        features, candidate_tokens = self.encode_candidates(
            candidate_mean,
            candidate_variance,
            best_objectives,
            progress,
            objective_mask=objective_mask,
        )
        batch_size, n_candidates, _ = candidate_tokens.shape
        mask = self._prepare_candidate_mask(
            candidate_mask,
            batch_size=batch_size,
            n_candidates=n_candidates,
            device=candidate_tokens.device,
        )
        history = self._prepare_history(
            history_mean=history_mean,
            history_variance=history_variance,
            history_best_objectives=history_best_objectives,
            history_progress=history_progress,
            history_rewards=history_rewards,
            history_q_values=history_q_values,
            history_objective_mask=history_objective_mask,
            history_mask=history_mask,
            batch_size=batch_size,
            device=candidate_tokens.device,
            dtype=candidate_tokens.dtype,
        )
        q_values = self._score_candidates(
            candidate_tokens=candidate_tokens,
            history=history,
            candidate_mask=mask,
        )
        ranking, policy_probs = self._decode(
            q_values,
            mask,
            decode_type=str(decode_type),
            epsilon=self.epsilon if epsilon is None else float(epsilon),
            temperature=self.temperature if temperature is None else float(temperature),
        )
        return {
            "observation_action_features": features,
            "candidate_tokens": candidate_tokens,
            "q_values": q_values,
            "logits": q_values,
            "policy_probs": policy_probs,
            "ranking": ranking,
            "action": ranking[:, 0],
            "history_mask": history["mask"],
        }

    def load_transformer_weights_from(self, other: "BOFormer") -> None:
        """Transfer objective-count-independent weights from another BOFormer."""
        if self.hidden_dim != other.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: destination={self.hidden_dim}, source={other.hidden_dim}."
            )
        self.transformer.load_state_dict(other.transformer.state_dict(), strict=True)
        self.reward_embedding.load_state_dict(other.reward_embedding.state_dict(), strict=True)
        self.q_value_embedding.load_state_dict(other.q_value_embedding.state_dict(), strict=True)
        self.q_head.load_state_dict(other.q_head.state_dict(), strict=True)


__all__ = ["BOFormer", "upgrade_legacy_observation_action_weight"]
