import torch

from agents.base import BaseAgent, TsAttnBlock
from agents.dueling_q import DuelingQDecoder


class _DiscDecodeMixin:
    def _prepare_progress_for_candidates(self, h_surr, progress):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        return progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

    def _actor_logits(self, h_surr, progress, candidate_mask=None):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        progress_cand = progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

        q_values = self.q_decoder(
            h_cand=h_surr,
            aux_cand=progress_cand,
            aux_state=progress_state,
            candidate_mask=candidate_mask,
        )

        return q_values

    def _epsilon_greedy_ranking(self, q_values, epsilon=0.05, candidate_mask=None):
        """
        Return ranking [B, N] by epsilon-greedy.

        With probability epsilon:
            random valid ranking
        Otherwise:
            descending Q ranking
        """
        bsz, n_cand = q_values.shape
        device = q_values.device

        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask, -1e9)

        greedy_ranking = torch.argsort(q_values, dim=-1, descending=True)

        random_rankings = []
        for b in range(bsz):
            if candidate_mask is None:
                perm = torch.randperm(n_cand, device=device)
            else:
                valid_idx = torch.nonzero(candidate_mask[b], as_tuple=False).squeeze(-1)
                invalid_idx = torch.nonzero(~candidate_mask[b], as_tuple=False).squeeze(-1)

                valid_perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
                perm = torch.cat([valid_perm, invalid_idx], dim=0)

            random_rankings.append(perm)

        random_ranking = torch.stack(random_rankings, dim=0)

        use_random = torch.rand(bsz, device=device) < float(epsilon)

        ranking = torch.where(
            use_random.unsqueeze(1),
            random_ranking,
            greedy_ranking,
        )

        return ranking

    def decode_ranking(
        self,
        h_surr,
        progress,
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
        candidate_mask=None,
        epsilon=None,
    ):
        h_surr = self._ensure_batch(h_surr)

        q_values = self._actor_logits(
            h_surr=h_surr,
            progress=progress,
            candidate_mask=candidate_mask,
        )

        if epsilon is None:
            epsilon = self.epsilon

        if target_ranking is not None:
            ranking = target_ranking

        elif decode_type == "epsilon_greedy":
            ranking = self._epsilon_greedy_ranking(
                q_values=q_values,
                epsilon=epsilon,
                candidate_mask=candidate_mask,
            )

        elif decode_type in ["q_greedy", "greedy"]:
            ranking = torch.argsort(q_values, dim=-1, descending=True)

        elif decode_type == "softmax_sample":
            decode_steps = q_values.size(1) if max_decode_steps is None else min(
                int(max_decode_steps), q_values.size(1)
            )
            ranking = self._sample_actions_without_replacement(q_values, decode_steps)

        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        if max_decode_steps is not None:
            ranking = ranking[:, : min(int(max_decode_steps), ranking.size(1))]

        return {
            "logits": q_values,
            "q_values": q_values,
            "ranking": ranking,
        }

    def _sample_actions_without_replacement(self, logits, k):
        bsz, n_sur = logits.shape
        selected_mask = torch.zeros(bsz, n_sur, dtype=torch.bool, device=logits.device)
        actions = []
        decode_steps = min(int(k), n_sur)

        for _ in range(decode_steps):
            masked_logits = logits.masked_fill(selected_mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            chosen_idx = dist.sample()
            actions.append(chosen_idx.unsqueeze(1))
            selected_mask = selected_mask.scatter(1, chosen_idx.unsqueeze(1), True)

        return torch.cat(actions, dim=1)

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
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
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

        decoded = self.decode_ranking(
            h_surr=encoded["H_surr"],
            progress=encoded["progress"],
            target_ranking=target_ranking,
            decode_type=decode_type,
            max_decode_steps=max_decode_steps,
            candidate_mask=candidate_mask,
            epsilon=epsilon,
        )

        encoded.update(decoded)
        return encoded


class Disc(_DiscDecodeMixin, BaseAgent):
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
        )
        self.epsilon = float(epsilon)
        self.q_decoder = DuelingQDecoder(
            hidden_dim=hidden_dim,
            aux_dim=1,
            dropout=dropout,
            logit_scale=logit_scale,
            value_uses_embedding=value_uses_embedding,
        )
        self.true_objective_attn = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.surr_objective_attn = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)

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
        m_true, m_surr, dim_mask, archive_mask, candidate_mask = self._prepare_inputs(
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

        e_true = self.W_true(m_true)
        e_surr = self.W_surr(m_surr)

        s_true = self.encoder_true(e_true, dim_mask=dim_mask, individual_mask=archive_mask)
        s_surr = self.encoder_surr(e_surr, dim_mask=dim_mask, individual_mask=candidate_mask)

        bsz, n_obj, n_true, hidden_dim = s_true.shape
        n_surr = int(s_surr.shape[2])
        s_true_obj = s_true.permute(0, 2, 1, 3).reshape(bsz * n_true, n_obj, hidden_dim)
        s_surr_obj = s_surr.permute(0, 2, 1, 3).reshape(bsz * n_surr, n_obj, hidden_dim)
        s_true_obj = self.true_objective_attn(s_true_obj)
        s_surr_obj = self.surr_objective_attn(s_surr_obj)
        s_true = s_true_obj.reshape(bsz, n_true, n_obj, hidden_dim).permute(0, 2, 1, 3)
        s_surr = s_surr_obj.reshape(bsz, n_surr, n_obj, hidden_dim).permute(0, 2, 1, 3)

        h_true = s_true.mean(dim=1)
        h_surr_raw = s_surr.mean(dim=1)
        h_surr = self.cross_space_attn(
            h_surr_raw,
            h_true,
            h_true,
            key_padding_mask=(~archive_mask),
        )

        progress = self._prepare_progress(
            progress=progress,
            device=h_true.device,
            dtype=h_true.dtype,
            batch_size=h_true.size(0),
        )

        return {
            "H_true": h_true,
            "H_surr": h_surr,
            "progress": progress,
        }


class DiscAF(BaseAgent):
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
        )
        self.epsilon = float(epsilon)
        self.q_decoder = DuelingQDecoder(
            hidden_dim=hidden_dim,
            aux_dim=1,
            dropout=dropout,
            logit_scale=logit_scale,
            value_uses_embedding=value_uses_embedding,
        )
        self.true_objective_attn = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.surr_objective_attn = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)

    def _aggregate_surrogate_embeddings(self, e_surr, dim_mask):
        e_surr = self._ensure_batch(e_surr)
        if dim_mask is None:
            dim_agg = e_surr.mean(dim=2)
        else:
            dim_mask_f = dim_mask.to(device=e_surr.device, dtype=e_surr.dtype).view(
                e_surr.size(0), 1, e_surr.size(2), 1, 1
            )
            dim_sum = (e_surr * dim_mask_f).sum(dim=2)
            dim_denom = dim_mask_f.sum(dim=2).clamp_min(1.0)
            dim_agg = dim_sum / dim_denom
        return dim_agg.mean(dim=1)

    def _encode_surrogate_without_candidate_attention(self, e_surr, dim_mask):
        """Encode candidate dimensions/objectives without mixing candidates."""
        e_surr = self._ensure_batch(e_surr)
        bsz, n_obj, n_dim, n_surr, hidden_dim = e_surr.shape

        x = e_surr.permute(0, 1, 3, 2, 4).reshape(bsz * n_obj * n_surr, n_dim, hidden_dim)
        x = self.encoder_surr.position(x)
        dim_padding_mask = None
        if dim_mask is not None:
            dim_mask = dim_mask.to(device=x.device, dtype=torch.bool)
            dim_padding_mask = (~dim_mask).unsqueeze(1).unsqueeze(1)
            dim_padding_mask = dim_padding_mask.expand(-1, n_obj, n_surr, -1)
            dim_padding_mask = dim_padding_mask.reshape(bsz * n_obj * n_surr, n_dim)
        x = self.encoder_surr.cross_dimension(x, key_padding_mask=dim_padding_mask)
        x = x.reshape(bsz, n_obj, n_surr, n_dim, hidden_dim)

        if dim_mask is None:
            return x.mean(dim=3)

        dim_mask_expanded = dim_mask.to(device=x.device, dtype=x.dtype).view(bsz, 1, 1, n_dim, 1)
        masked_sum = (x * dim_mask_expanded).sum(dim=3)
        denom = dim_mask_expanded.sum(dim=3).clamp_min(1.0)
        return masked_sum / denom

    def _prepare_progress_for_candidates(self, h_surr, progress):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        return progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

    def _actor_logits(self, h_surr, progress, candidate_mask=None):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        progress_cand = progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

        q_values = self.q_decoder(
            h_cand=h_surr,
            aux_cand=progress_cand,
            aux_state=progress_state,
            candidate_mask=candidate_mask,
        )

        return q_values

    def _epsilon_greedy_ranking(self, q_values, epsilon=0.05, candidate_mask=None):
        bsz, n_cand = q_values.shape
        device = q_values.device

        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask, -1e9)

        greedy_ranking = torch.argsort(q_values, dim=-1, descending=True)

        random_rankings = []
        for b in range(bsz):
            if candidate_mask is None:
                perm = torch.randperm(n_cand, device=device)
            else:
                valid_idx = torch.nonzero(candidate_mask[b], as_tuple=False).squeeze(-1)
                invalid_idx = torch.nonzero(~candidate_mask[b], as_tuple=False).squeeze(-1)
                valid_perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
                perm = torch.cat([valid_perm, invalid_idx], dim=0)
            random_rankings.append(perm)

        random_ranking = torch.stack(random_rankings, dim=0)
        use_random = torch.rand(bsz, device=device) < float(epsilon)
        ranking = torch.where(use_random.unsqueeze(1), random_ranking, greedy_ranking)
        return ranking

    def decode_ranking(
        self,
        h_surr,
        progress,
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
        candidate_mask=None,
        epsilon=None,
    ):
        h_surr = self._ensure_batch(h_surr)

        q_values = self._actor_logits(
            h_surr=h_surr,
            progress=progress,
            candidate_mask=candidate_mask,
        )

        if epsilon is None:
            epsilon = self.epsilon

        if target_ranking is not None:
            ranking = target_ranking
        elif decode_type == "epsilon_greedy":
            ranking = self._epsilon_greedy_ranking(
                q_values=q_values,
                epsilon=epsilon,
                candidate_mask=candidate_mask,
            )
        elif decode_type in ["q_greedy", "greedy"]:
            ranking = torch.argsort(q_values, dim=-1, descending=True)
        elif decode_type == "softmax_sample":
            decode_steps = q_values.size(1) if max_decode_steps is None else min(
                int(max_decode_steps), q_values.size(1)
            )
            ranking = self._sample_actions_without_replacement(q_values, decode_steps)
        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        if max_decode_steps is not None:
            ranking = ranking[:, : min(int(max_decode_steps), ranking.size(1))]

        return {
            "logits": q_values,
            "q_values": q_values,
            "ranking": ranking,
        }

    def _sample_actions_without_replacement(self, logits, k):
        bsz, n_sur = logits.shape
        selected_mask = torch.zeros(bsz, n_sur, dtype=torch.bool, device=logits.device)
        actions = []
        decode_steps = min(int(k), n_sur)

        for _ in range(decode_steps):
            masked_logits = logits.masked_fill(selected_mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            chosen_idx = dist.sample()
            actions.append(chosen_idx.unsqueeze(1))
            selected_mask = selected_mask.scatter(1, chosen_idx.unsqueeze(1), True)

        return torch.cat(actions, dim=1)

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
        m_true, m_surr, dim_mask, archive_mask, candidate_mask = self._prepare_inputs(
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

        e_true = self.W_true(m_true)
        e_surr = self.W_surr(m_surr)

        s_true = self.encoder_true(e_true, dim_mask=dim_mask, individual_mask=archive_mask)
        bsz, n_obj, n_true, hidden_dim = s_true.shape
        s_true_obj = s_true.permute(0, 2, 1, 3).reshape(bsz * n_true, n_obj, hidden_dim)
        s_true_obj = self.true_objective_attn(s_true_obj)
        s_true = s_true_obj.reshape(bsz, n_true, n_obj, hidden_dim).permute(0, 2, 1, 3)
        h_true = s_true.mean(dim=1)

        s_surr = self._encode_surrogate_without_candidate_attention(e_surr, dim_mask)
        n_surr = int(s_surr.shape[2])
        s_surr_obj = s_surr.permute(0, 2, 1, 3).reshape(bsz * n_surr, n_obj, hidden_dim)
        s_surr_obj = self.surr_objective_attn(s_surr_obj)
        s_surr = s_surr_obj.reshape(bsz, n_surr, n_obj, hidden_dim).permute(0, 2, 1, 3)
        h_surr_raw = s_surr.mean(dim=1)
        h_surr = self.cross_space_attn(
            h_surr_raw,
            h_true,
            h_true,
            key_padding_mask=(~archive_mask),
        )

        progress = self._prepare_progress(
            progress=progress,
            device=h_true.device,
            dtype=h_true.dtype,
            batch_size=h_true.size(0),
        )

        return {
            "H_true": h_true,
            "H_surr": h_surr,
            "progress": progress,
        }

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
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
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

        decoded = self.decode_ranking(
            h_surr=encoded["H_surr"],
            progress=encoded["progress"],
            target_ranking=target_ranking,
            decode_type=decode_type,
            max_decode_steps=max_decode_steps,
            candidate_mask=candidate_mask,
            epsilon=epsilon,
        )

        encoded.update(decoded)
        return encoded


class DiscAF2(_DiscDecodeMixin, BaseAgent):
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
        )
        self.epsilon = float(epsilon)
        self.q_decoder = DuelingQDecoder(
            hidden_dim=hidden_dim,
            aux_dim=1,
            dropout=dropout,
            logit_scale=logit_scale,
            value_uses_embedding=value_uses_embedding,
        )
        self.surr_objective_attn = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)

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
        _, m_surr, dim_mask, _, candidate_mask = self._prepare_inputs(
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

        e_surr = self.W_surr(m_surr)
        s_surr = self.encoder_surr(e_surr, dim_mask=dim_mask, individual_mask=candidate_mask)

        bsz, n_obj, n_surr, hidden_dim = s_surr.shape
        s_surr_obj = s_surr.permute(0, 2, 1, 3).reshape(bsz * n_surr, n_obj, hidden_dim)
        s_surr_obj = self.surr_objective_attn(s_surr_obj)
        s_surr = s_surr_obj.reshape(bsz, n_surr, n_obj, hidden_dim).permute(0, 2, 1, 3)
        h_surr = s_surr.mean(dim=1)

        progress = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        return {
            "H_true": None,
            "H_surr": h_surr,
            "progress": progress,
        }
