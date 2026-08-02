import torch
import torch.nn as nn

from agents.disc import Disc


class SingleQDecoder(nn.Module):
    """Predict candidate Q-values with one shared MLP instead of dueling heads."""

    def __init__(
        self,
        hidden_dim=128,
        aux_dim=1,
        dropout=0.0,
        logit_scale=5.0,
    ):
        super().__init__()
        self.logit_scale = float(logit_scale)
        input_dim = int(hidden_dim) + int(aux_dim)
        self.q_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward(self, h_cand, aux_cand, aux_state=None, candidate_mask=None):
        del aux_state
        q_input = torch.cat([h_cand, aux_cand], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)
        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask, -1e9)
        return self.logit_scale * q_values


class DiscSingleDQN(Disc):
    """Original DISC encoder with a non-dueling, candidate-wise Q-network."""

    def __init__(
        self,
        hidden_dim=128,
        n_heads=8,
        ff_dim=256,
        dropout=0.0,
        logit_scale=5.0,
        value_uses_embedding=True,
        epsilon=0.05,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            logit_scale=logit_scale,
            value_uses_embedding=value_uses_embedding,
            epsilon=epsilon,
        )
        self.q_decoder = SingleQDecoder(
            hidden_dim=hidden_dim,
            aux_dim=1,
            dropout=dropout,
            logit_scale=logit_scale,
        )
