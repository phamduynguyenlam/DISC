import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=2048):
        super().__init__()
        position = np.arange(int(max_len), dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, int(hidden_dim), 2, dtype=np.float32)
            * (-np.log(10000.0) / float(hidden_dim))
        ).astype(np.float32, copy=False)
        pe = np.zeros((int(max_len), int(hidden_dim)), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe_tensor = torch.from_numpy(pe).unsqueeze(0)
        self.register_buffer("pe", pe_tensor, persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TsAttnBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class LandscapeEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.cross_individual = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_dimension = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.position = PositionalEncoding(hidden_dim)

    def forward(self, x, dim_mask=None, individual_mask=None):
        bsz, n_obj, n_dim, n_ind, hidden_dim = x.shape

        x = x.reshape(bsz * n_obj * n_dim, n_ind, hidden_dim)
        individual_padding_mask = None
        if individual_mask is not None:
            individual_mask = individual_mask.to(device=x.device, dtype=torch.bool)
            individual_padding_mask = (~individual_mask).unsqueeze(1).unsqueeze(1)
            individual_padding_mask = individual_padding_mask.expand(-1, n_obj, n_dim, -1)
            individual_padding_mask = individual_padding_mask.reshape(bsz * n_obj * n_dim, n_ind)
        x = self.cross_individual(x, key_padding_mask=individual_padding_mask)
        x = x.reshape(bsz, n_obj, n_dim, n_ind, hidden_dim)

        x = x.permute(0, 1, 3, 2, 4).reshape(bsz * n_obj * n_ind, n_dim, hidden_dim)
        x = self.position(x)
        dim_padding_mask = None
        if dim_mask is not None:
            dim_mask = dim_mask.to(device=x.device, dtype=torch.bool)
            dim_padding_mask = (~dim_mask).unsqueeze(1).unsqueeze(1)
            dim_padding_mask = dim_padding_mask.expand(-1, n_obj, n_ind, -1)
            dim_padding_mask = dim_padding_mask.reshape(bsz * n_obj * n_ind, n_dim)
        x = self.cross_dimension(x, key_padding_mask=dim_padding_mask)
        x = x.reshape(bsz, n_obj, n_ind, n_dim, hidden_dim)

        if dim_mask is None:
            return x.mean(dim=3)

        dim_mask_expanded = dim_mask.to(device=x.device, dtype=x.dtype).view(bsz, 1, 1, n_dim, 1)
        masked_sum = (x * dim_mask_expanded).sum(dim=3)
        denom = dim_mask_expanded.sum(dim=3).clamp_min(1.0)
        return masked_sum / denom


class CrossSpaceAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_out, _ = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(query + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class BaseAgent(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        n_heads=8,
        ff_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_true = nn.Linear(2, hidden_dim)
        self.W_surr = nn.Linear(3, hidden_dim)
        self.encoder_true = LandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.encoder_surr = LandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_space_attn = CrossSpaceAttention(hidden_dim, n_heads, ff_dim, dropout)

    @staticmethod
    def _ensure_batch(x):
        return x.unsqueeze(0) if x.dim() == 2 else x

    @staticmethod
    def _normalize_by_range(x, lower, upper):
        denom = (upper - lower).clamp_min(1e-12)
        return ((x - lower) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_by_extrema(*tensors):
        stacked = torch.cat(tensors, dim=1)
        min_v = stacked.amin(dim=1, keepdim=True)
        max_v = stacked.amax(dim=1, keepdim=True)
        denom = (max_v - min_v).clamp_min(1e-12)
        return tuple(((tensor - min_v) / denom).clamp(0.0, 1.0) for tensor in tensors)

    @staticmethod
    def _prepare_progress(
        progress,
        device,
        dtype,
        batch_size,
    ):
        progress = torch.as_tensor(
            progress,
            device=device,
            dtype=dtype,
        )

        if progress.dim() == 0:
            progress = progress.repeat(batch_size)

        progress = progress.reshape(batch_size, -1)

        if progress.size(1) != 1:
            progress = progress[:, :1]

        return progress.clamp(0.0, 1.0)

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
            elif archive_mask.dim() == 2:
                if archive_mask.size(0) != batch_size:
                    raise ValueError(f"archive_mask batch size mismatch: {archive_mask.size(0)} vs {batch_size}")
            else:
                raise ValueError(f"Unsupported archive_mask shape: {tuple(archive_mask.shape)}")
            if archive_mask.size(1) != n_archive:
                raise ValueError(f"archive_mask width mismatch: {archive_mask.size(1)} vs {n_archive}")

        if candidate_mask is not None:
            candidate_mask = torch.as_tensor(candidate_mask, device=device, dtype=torch.bool)
            if candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.view(1, -1).expand(batch_size, -1)
            elif candidate_mask.dim() == 2:
                if candidate_mask.size(0) != batch_size:
                    raise ValueError(f"candidate_mask batch size mismatch: {candidate_mask.size(0)} vs {batch_size}")
            else:
                raise ValueError(f"Unsupported candidate_mask shape: {tuple(candidate_mask.shape)}")
            if candidate_mask.size(1) != n_candidates:
                raise ValueError(f"candidate_mask width mismatch: {candidate_mask.size(1)} vs {n_candidates}")

        lower = torch.as_tensor(lower_bound, device=device, dtype=dtype)
        upper = torch.as_tensor(upper_bound, device=device, dtype=dtype)

        if lower.dim() == 0:
            lower = lower.repeat(n_dim).view(1, n_dim).expand(batch_size, -1)
        elif lower.dim() == 1:
            lower = lower.view(1, -1).expand(batch_size, -1)
        elif lower.dim() == 2:
            if lower.size(0) != batch_size:
                raise ValueError(f"lower_bound batch size mismatch: {lower.size(0)} vs {batch_size}")
        else:
            raise ValueError(f"Unsupported lower_bound shape: {tuple(lower.shape)}")

        if upper.dim() == 0:
            upper = upper.repeat(n_dim).view(1, n_dim).expand(batch_size, -1)
        elif upper.dim() == 1:
            upper = upper.view(1, -1).expand(batch_size, -1)
        elif upper.dim() == 2:
            if upper.size(0) != batch_size:
                raise ValueError(f"upper_bound batch size mismatch: {upper.size(0)} vs {batch_size}")
        else:
            raise ValueError(f"Unsupported upper_bound shape: {tuple(upper.shape)}")

        dim_mask = (upper - lower).abs() > 1e-12

        lower = lower.unsqueeze(1)
        upper = upper.unsqueeze(1)

        x_true = self._normalize_by_range(x_true, lower, upper)
        x_sur = self._normalize_by_range(x_sur, lower, upper)
        y_true, y_sur = self._normalize_by_extrema(y_true, y_sur)
        (sigma_sur,) = self._normalize_by_extrema(sigma_sur)

        x_true_expand = x_true.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_true_expand = x_true_expand.expand(-1, y_true.size(-1), -1, -1, -1)
        y_true_expand = y_true.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_true_expand = y_true_expand.expand(-1, -1, x_true.size(-1), -1, -1)
        m_true = torch.cat((x_true_expand, y_true_expand), dim=-1)

        x_sur_expand = x_sur.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_sur_expand = x_sur_expand.expand(-1, y_sur.size(-1), -1, -1, -1)
        y_sur_expand = y_sur.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_sur_expand = y_sur_expand.expand(-1, -1, x_sur.size(-1), -1, -1)
        sigma_expand = sigma_sur.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        sigma_expand = sigma_expand.expand(-1, -1, x_sur.size(-1), -1, -1)
        m_surr = torch.cat((x_sur_expand, y_sur_expand, sigma_expand), dim=-1)

        if archive_mask is None:
            archive_mask = torch.ones(
                x_true.size(0),
                x_true.size(1),
                device=device,
                dtype=torch.bool,
            )
        if candidate_mask is None:
            candidate_mask = torch.ones(
                x_sur.size(0),
                x_sur.size(1),
                device=device,
                dtype=torch.bool,
            )

        return m_true, m_surr, dim_mask, archive_mask, candidate_mask

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
