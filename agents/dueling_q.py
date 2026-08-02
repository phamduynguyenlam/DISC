import torch
import torch.nn as nn


class DuelingQDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        aux_dim=1,
        dropout=0.0,
        logit_scale=5.0,
        value_uses_embedding=True,
    ):
        super().__init__()

        self.logit_scale = float(logit_scale)
        self.value_uses_embedding = bool(value_uses_embedding)

        self.advantage_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + aux_dim),
            nn.Linear(hidden_dim + aux_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        value_in_dim = hidden_dim + aux_dim if self.value_uses_embedding else aux_dim

        self.value_head = nn.Sequential(
            nn.LayerNorm(value_in_dim),
            nn.Linear(value_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _masked_mean(self, x, mask=None, dim=1):
        if mask is None:
            return x.mean(dim=dim)

        mask_f = mask.to(dtype=x.dtype)
        while mask_f.dim() < x.dim():
            mask_f = mask_f.unsqueeze(-1)

        denom = mask_f.sum(dim=dim).clamp_min(1.0)
        return (x * mask_f).sum(dim=dim) / denom

    def forward(self, h_cand, aux_cand, aux_state=None, candidate_mask=None):
        """
        h_cand: [B, N, H]
        aux_cand: [B, N, A], e.g. progress expanded per candidate
        aux_state: [B, A], optional. If None, use aux_cand[:, 0, :]
        candidate_mask: [B, N] bool, optional

        return:
            q_values: [B, N]
        """
        if aux_state is None:
            aux_state = aux_cand[:, 0, :]

        adv_input = torch.cat([h_cand, aux_cand], dim=-1)
        advantage = self.advantage_head(adv_input).squeeze(-1)  # [B, N]

        if self.value_uses_embedding:
            h_global = self._masked_mean(h_cand, mask=candidate_mask, dim=1)
            value_input = torch.cat([h_global, aux_state], dim=-1)
        else:
            value_input = aux_state

        value = self.value_head(value_input)  # [B, 1]

        if candidate_mask is not None:
            adv_for_mean = advantage.masked_fill(~candidate_mask, 0.0)
            denom = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1)
            adv_mean = adv_for_mean.sum(dim=1, keepdim=True) / denom
        else:
            adv_mean = advantage.mean(dim=1, keepdim=True)

        q_values = value + advantage - adv_mean

        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask, -1e9)

        return q_values * self.logit_scale