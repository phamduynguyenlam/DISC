from __future__ import annotations

import torch
import torch.nn as nn

from agents.base import LandscapeEncoder, TsAttnBlock
from infill import NDA, EPDIExploitation, EPDIExploration, NDPBIConvergence, NDPBIDiversity


class DuelingActionDecoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 16, n_actions: int = 6, dropout: float = 0.0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)

        self.advantage_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.n_actions),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"DuelingActionDecoder expects [B, D], got {tuple(z.shape)}.")
        advantage = self.advantage_head(z)
        value = self.value_head(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DBSAEAAgent(nn.Module):
    ACTION_NAMES = (
        "regenerate_offspring",
        "nd_a",
        "nd_pbi_convergence",
        "nd_pbi_diversity",
        "epdi_exploitation",
        "epdi_exploration",
    )

    def __init__(
        self,
        hidden_dim: int = 16,
        ff_dim: int = 256,
        dropout: float = 0.0,
        logit_scale: float = 5.0,
        epsilon: float = 0.05,
        decoder_hidden_dim: int = 16,
        n_heads: int = 1,
        **_: object,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.logit_scale = float(logit_scale)
        self.epsilon = float(epsilon)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})."
            )

        self.W_true = nn.Linear(2, self.hidden_dim)
        self.W_surr = nn.Linear(3, self.hidden_dim)
        self.encoder_true = LandscapeEncoder(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.encoder_surr = LandscapeEncoder(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.stage2_true_cross_individual = TsAttnBlock(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.stage2_true_cross_objective = TsAttnBlock(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.stage2_surr_cross_individual = TsAttnBlock(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.stage2_surr_cross_objective = TsAttnBlock(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.q_decoder = DuelingActionDecoder(
            state_dim=2 * self.hidden_dim + 1,
            hidden_dim=int(decoder_hidden_dim),
            n_actions=len(self.ACTION_NAMES),
            dropout=self.dropout,
        )

        self.infill_criteria = {
            1: NDA(),
            2: NDPBIConvergence(),
            3: NDPBIDiversity(),
            4: EPDIExploitation(),
            5: EPDIExploration(),
        }

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 2 else x

    @staticmethod
    def _normalize_by_range(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        denom = (upper - lower).clamp_min(1e-12)
        return ((x - lower) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _prepare_progress(progress, device, dtype, batch_size: int) -> torch.Tensor:
        progress_tensor = torch.as_tensor(progress, device=device, dtype=dtype)
        if progress_tensor.dim() == 0:
            progress_tensor = progress_tensor.repeat(int(batch_size))
        progress_tensor = progress_tensor.reshape(int(batch_size), -1)
        if progress_tensor.size(1) != 1:
            progress_tensor = progress_tensor[:, :1]
        return progress_tensor.clamp(0.0, 1.0)

    def _prepare_inputs(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        lower_bound,
        upper_bound,
        archive_mask=None,
        candidate_mask=None,
    ):
        x_true = self._ensure_batch(x_true).float()
        y_true = self._ensure_batch(y_true).float()
        x_sur = self._ensure_batch(x_sur).float()
        y_sur = self._ensure_batch(y_sur).float()
        sigma_sur = self._ensure_batch(sigma_sur).float()

        device = x_true.device
        dtype = x_true.dtype
        batch_size = x_true.size(0)
        n_dim = x_true.size(-1)
        n_archive = x_true.size(1)
        n_candidates = x_sur.size(1)

        if archive_mask is not None:
            archive_mask = torch.as_tensor(archive_mask, device=device, dtype=torch.bool)
            if archive_mask.dim() == 1:
                archive_mask = archive_mask.view(1, -1).expand(batch_size, -1)
            if archive_mask.size(1) != n_archive:
                raise ValueError(f"archive_mask width mismatch: {archive_mask.size(1)} vs {n_archive}")

        if candidate_mask is not None:
            candidate_mask = torch.as_tensor(candidate_mask, device=device, dtype=torch.bool)
            if candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.view(1, -1).expand(batch_size, -1)
            if candidate_mask.size(1) != n_candidates:
                raise ValueError(f"candidate_mask width mismatch: {candidate_mask.size(1)} vs {n_candidates}")

        lower = torch.as_tensor(lower_bound, device=device, dtype=dtype)
        upper = torch.as_tensor(upper_bound, device=device, dtype=dtype)

        if lower.dim() == 0:
            lower = lower.repeat(n_dim).view(1, n_dim).expand(batch_size, -1)
        elif lower.dim() == 1:
            lower = lower.view(1, -1).expand(batch_size, -1)
        if upper.dim() == 0:
            upper = upper.repeat(n_dim).view(1, n_dim).expand(batch_size, -1)
        elif upper.dim() == 1:
            upper = upper.view(1, -1).expand(batch_size, -1)

        dim_mask = (upper - lower).abs() > 1e-12
        lower = lower.unsqueeze(1)
        upper = upper.unsqueeze(1)

        x_true_norm = self._normalize_by_range(x_true, lower, upper)
        x_sur_norm = self._normalize_by_range(x_sur, lower, upper)

        stacked_y = torch.cat([y_true, y_sur], dim=1)
        y_min = stacked_y.amin(dim=1, keepdim=True)
        y_max = stacked_y.amax(dim=1, keepdim=True)
        y_true_norm = ((y_true - y_min) / (y_max - y_min).clamp_min(1e-12)).clamp(0.0, 1.0)
        y_sur_norm = ((y_sur - y_min) / (y_max - y_min).clamp_min(1e-12)).clamp(0.0, 1.0)
        sigma_sur_norm = sigma_sur / sigma_sur.amax(dim=1, keepdim=True).clamp_min(1e-12)

        x_true_expand = x_true_norm.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_true_expand = x_true_expand.expand(-1, y_true_norm.size(-1), -1, -1, -1)
        y_true_expand = y_true_norm.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_true_expand = y_true_expand.expand(-1, -1, x_true_norm.size(-1), -1, -1)
        e_true = torch.cat((x_true_expand, y_true_expand), dim=-1)

        x_sur_expand = x_sur_norm.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_sur_expand = x_sur_expand.expand(-1, y_sur_norm.size(-1), -1, -1, -1)
        y_sur_expand = y_sur_norm.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_sur_expand = y_sur_expand.expand(-1, -1, x_sur_norm.size(-1), -1, -1)
        sigma_expand = sigma_sur_norm.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        sigma_expand = sigma_expand.expand(-1, -1, x_sur_norm.size(-1), -1, -1)
        e_surr = torch.cat((x_sur_expand, y_sur_expand, sigma_expand), dim=-1)

        if archive_mask is None:
            archive_mask = torch.ones(batch_size, n_archive, device=device, dtype=torch.bool)
        if candidate_mask is None:
            candidate_mask = torch.ones(batch_size, n_candidates, device=device, dtype=torch.bool)

        return {
            "E_true": e_true,
            "E_surr": e_surr,
            "dim_mask": dim_mask,
            "archive_mask": archive_mask,
            "candidate_mask": candidate_mask,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        }

    @staticmethod
    def _masked_pool_over_individuals(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=2)
        mask_f = mask.to(device=h.device, dtype=h.dtype).view(h.size(0), 1, h.size(2), 1)
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        return (h * mask_f).sum(dim=2) / denom

    def _stage_two_refine(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        cross_individual_block: TsAttnBlock,
        cross_objective_block: TsAttnBlock,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_obj, n_individuals, hidden_dim = h.shape

        stage = h.reshape(batch_size * n_obj, n_individuals, hidden_dim)
        individual_padding_mask = None
        if mask is not None:
            individual_padding_mask = (~mask.to(device=h.device, dtype=torch.bool)).unsqueeze(1)
            individual_padding_mask = individual_padding_mask.expand(-1, n_obj, -1)
            individual_padding_mask = individual_padding_mask.reshape(batch_size * n_obj, n_individuals)
        stage = cross_individual_block(stage, key_padding_mask=individual_padding_mask)
        stage = stage.reshape(batch_size, n_obj, n_individuals, hidden_dim)

        stage = stage.permute(0, 2, 1, 3).reshape(batch_size * n_individuals, n_obj, hidden_dim)
        stage = cross_objective_block(stage)
        stage = stage.reshape(batch_size, n_individuals, n_obj, hidden_dim).permute(0, 2, 1, 3)

        pooled = self._masked_pool_over_individuals(stage, mask).mean(dim=1)
        return stage, pooled

    def encode(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        archive_mask=None,
        candidate_mask=None,
    ):
        prepared = self._prepare_inputs(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=archive_mask,
            candidate_mask=candidate_mask,
        )

        h_true = self.encoder_true(
            self.W_true(prepared["E_true"]),
            dim_mask=prepared["dim_mask"],
            individual_mask=prepared["archive_mask"],
        )
        h_surr = self.encoder_surr(
            self.W_surr(prepared["E_surr"]),
            dim_mask=prepared["dim_mask"],
            individual_mask=prepared["candidate_mask"],
        )

        h_true_stage2, h_true_pool = self._stage_two_refine(
            h_true,
            prepared["archive_mask"],
            cross_individual_block=self.stage2_true_cross_individual,
            cross_objective_block=self.stage2_true_cross_objective,
        )
        h_surr_stage2, h_surr_pool = self._stage_two_refine(
            h_surr,
            prepared["candidate_mask"],
            cross_individual_block=self.stage2_surr_cross_individual,
            cross_objective_block=self.stage2_surr_cross_objective,
        )
        progress_state = self._prepare_progress(
            progress=progress,
            device=prepared["device"],
            dtype=prepared["dtype"],
            batch_size=prepared["batch_size"],
        )
        z = torch.cat([h_true_pool, h_surr_pool, progress_state], dim=-1)
        return {
            "H_true": h_true,
            "H_surr": h_surr,
            "H_true_stage2": h_true_stage2,
            "H_surr_stage2": h_surr_stage2,
            "h_true_pool": h_true_pool,
            "h_surr_pool": h_surr_pool,
            "progress": progress_state,
            "z": z,
        }

    def _epsilon_greedy_action(self, q_values: torch.Tensor, epsilon: float) -> torch.Tensor:
        batch_size, n_actions = q_values.shape
        greedy = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(0, n_actions, (batch_size,), device=q_values.device)
        use_random = torch.rand(batch_size, device=q_values.device) < float(epsilon)
        return torch.where(use_random, random_actions, greedy)

    def decode_action(self, z: torch.Tensor, decode_type: str = "epsilon_greedy", epsilon: float | None = None):
        q_values = self.q_decoder(z) * self.logit_scale
        if epsilon is None:
            epsilon = self.epsilon

        if decode_type in {"greedy", "q_greedy"}:
            action = torch.argmax(q_values, dim=1)
        elif decode_type == "softmax_sample":
            probs = torch.softmax(q_values, dim=-1)
            action = torch.distributions.Categorical(probs=probs).sample()
        elif decode_type == "epsilon_greedy":
            action = self._epsilon_greedy_action(q_values, float(epsilon))
        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        return {
            "logits": q_values,
            "q_values": q_values,
            "action": action,
        }

    def criterion_for_action(self, action_idx: int):
        idx = int(action_idx)
        if idx <= 0:
            return None
        return self.infill_criteria[idx]

    def select_candidate_from_action(
        self,
        *,
        action_idx: int,
        archive_y,
        candidate_mean,
        candidate_std,
        seed: int | None = None,
    ):
        criterion = self.criterion_for_action(int(action_idx))
        if criterion is None:
            return None, None
        selected_idx, scores = criterion.select_index(
            archive_y=archive_y,
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            seed=seed,
        )
        return int(selected_idx), scores

    def forward(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        archive_mask=None,
        candidate_mask=None,
        decode_type="epsilon_greedy",
        epsilon=None,
    ):
        encoded = self.encode(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            progress=progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=archive_mask,
            candidate_mask=candidate_mask,
        )
        decoded = self.decode_action(
            z=encoded["z"],
            decode_type=decode_type,
            epsilon=epsilon,
        )
        encoded.update(decoded)
        return encoded
