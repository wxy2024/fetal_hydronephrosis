import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x, lengths, delta=None, img_seq=None):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)

        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)

        return h_n[-1]


class CAGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, img_dim, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_mean = nn.Parameter(torch.zeros(input_dim))

        self.input_decay = nn.Linear(2, input_dim)
        self.hidden_decay = nn.Linear(2, hidden_dim)

        self.img_proj = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, hidden_dim),
            nn.GELU(),
        )

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _gamma(self, layer, decay_feat):
        return torch.exp(-torch.relu(layer(decay_feat)))

    def forward(self, x_t, h_prev, delta_t, img_t, img_prev):
        img_t_h = self.img_proj(img_t)
        img_prev_h = self.img_proj(img_prev)

        cos_sim = torch.cosine_similarity(img_t_h, img_prev_h, dim=-1)
        img_change = 1.0 - cos_sim
        img_change = torch.clamp(img_change, min=0.0, max=2.0)

        decay_feat = torch.stack([delta_t, img_change], dim=-1)

        gamma_x = self._gamma(self.input_decay, decay_feat)
        gamma_h = self._gamma(self.hidden_decay, decay_feat)

        x_hat = gamma_x * x_t + (1.0 - gamma_x) * self.x_mean.unsqueeze(0)
        h_hat = gamma_h * h_prev

        h_t = self.gru_cell(x_hat, h_hat)
        h_t = self.dropout(h_t)
        return h_t


class CAGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
        img_dim=None,
    ):
        super().__init__()

        if num_layers != 1:
            raise ValueError("CAGRUEncoder 当前建议使用 num_layers=1。")

        if img_dim is None:
            img_dim = input_dim

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.fw_cell = CAGRUCell(input_dim, hidden_dim, img_dim, dropout)

        if bidirectional:
            self.bw_cell = CAGRUCell(input_dim, hidden_dim, img_dim, dropout)

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def _run_one_direction(self, x, lengths, delta, img_seq, reverse=False):
        B, T, _ = x.shape
        device = x.device

        cell = self.bw_cell if reverse else self.fw_cell
        h = torch.zeros(B, self.hidden_dim, device=device)

        time_range = range(T - 1, -1, -1) if reverse else range(T)

        last_valid_h = h
        prev_img = img_seq[:, T - 1, :] if reverse else img_seq[:, 0, :]

        for t in time_range:
            active = lengths > t
            if not active.any():
                continue

            x_t = x[:, t, :]
            img_t = img_seq[:, t, :]
            delta_t = delta[:, t]

            h_new = cell(x_t, h, delta_t, img_t, prev_img)
            h = torch.where(active.unsqueeze(1), h_new, h)

            if not reverse:
                is_last = lengths == (t + 1)
                last_valid_h = torch.where(is_last.unsqueeze(1), h, last_valid_h)
            prev_img = img_t

        if reverse:
            return h

        return last_valid_h

    def forward(self, x, lengths, delta=None, img_seq=None):
        if delta is None or img_seq is None:
            raise ValueError("CAGRUEncoder 需要 delta 和 img_seq。")

        h_fw = self._run_one_direction(
            x=x,
            lengths=lengths,
            delta=delta,
            img_seq=img_seq,
            reverse=False,
        )

        if self.bidirectional:
            h_bw = self._run_one_direction(
                x=x,
                lengths=lengths,
                delta=delta,
                img_seq=img_seq,
                reverse=True,
            )
            return torch.cat([h_fw, h_bw], dim=1)

        return h_fw


class PSALMGRUDCell(nn.Module):
    """
    PSALM-GRU-D:
    Progression State-Aware Longitudinal Modeling + GRU-D-style decay.

    输入:
        x_t: 当前检查级特征
        x_prev: 上一次检查级特征
        h_prev: 上一步隐藏状态
        delta_t: 距离上一次检查的时间间隔

    核心:
        1. 显式估计相邻检查之间的 progression state:
           improved / stable / worsened
        2. progression state 参与输入调制和隐藏状态衰减
        3. 保留 GRU-D 的 delta-aware decay 思想
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=0.3,
        progression_dim=32,
        num_states=3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states

        self.x_mean = nn.Parameter(torch.zeros(input_dim))

        # progression state classifier
        # [x_t, x_prev, x_t - x_prev, |x_t - x_prev|, delta]
        prog_in_dim = input_dim * 4 + 1

        self.progression_mlp = nn.Sequential(
            nn.LayerNorm(prog_in_dim),
            nn.Linear(prog_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states),
        )

        # 把 progression state 概率映射为 embedding
        self.progression_embed = nn.Sequential(
            nn.Linear(num_states, progression_dim),
            nn.GELU(),
            nn.Linear(progression_dim, input_dim),
        )

        # progression-aware input gate
        self.input_gate = nn.Sequential(
            nn.Linear(num_states + 1, input_dim),
            nn.Sigmoid(),
        )

        # GRU-D-style decay
        # 这里不只看 delta，还看 progression state
        decay_in_dim = num_states + 1
        self.input_decay = nn.Linear(decay_in_dim, input_dim)
        self.hidden_decay = nn.Linear(decay_in_dim, hidden_dim)

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _gamma(self, layer, feat):
        return torch.exp(-torch.relu(layer(feat)))

    def forward(self, x_t, x_prev, h_prev, delta_t):
        # delta_t: [B]
        delta_t = delta_t.unsqueeze(-1)

        prog_feat = torch.cat(
            [
                x_t,
                x_prev,
                x_t - x_prev,
                torch.abs(x_t - x_prev),
                delta_t,
            ],
            dim=-1,
        )

        prog_logits = self.progression_mlp(prog_feat)
        prog_prob = F.softmax(prog_logits, dim=-1)

        decay_feat = torch.cat([delta_t, prog_prob], dim=-1)

        gamma_x = self._gamma(self.input_decay, decay_feat)
        gamma_h = self._gamma(self.hidden_decay, decay_feat)

        # GRU-D-style input decay
        x_decay = gamma_x * x_t + (1.0 - gamma_x) * self.x_mean.unsqueeze(0)

        # progression embedding
        prog_emb = self.progression_embed(prog_prob)

        # progression-aware input modulation
        gate = self.input_gate(decay_feat)
        x_hat = x_decay + gate * prog_emb

        # hidden state decay
        h_hat = gamma_h * h_prev

        h_t = self.gru_cell(x_hat, h_hat)
        h_t = self.dropout(h_t)

        return h_t, prog_prob


class PSALMGRUDEncoder(nn.Module):
    """
    Progression State-Aware Longitudinal Modeling Encoder.

    推荐先设置:
        GRU_BIDIRECTIONAL = False

    因为 progression state 本身有明确时间方向:
        visit t-1 -> visit t
    如果双向跑，反向分支虽然能训练，但临床解释性会变弱。
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=False,
        img_dim=None,
    ):
        super().__init__()

        if num_layers != 1:
            raise ValueError("PSALMGRUDEncoder 当前建议使用 num_layers=1。")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.fw_cell = PSALMGRUDCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        if bidirectional:
            self.bw_cell = PSALMGRUDCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def _run_one_direction(self, x, lengths, delta, reverse=False):
        B, T, _ = x.shape
        device = x.device

        cell = self.bw_cell if reverse else self.fw_cell
        h = torch.zeros(B, self.hidden_dim, device=device)

        time_range = range(T - 1, -1, -1) if reverse else range(T)

        last_valid_h = h

        if reverse:
            prev_x = x[:, T - 1, :]
        else:
            prev_x = x[:, 0, :]

        for t in time_range:
            active = lengths > t
            if not active.any():
                continue

            x_t = x[:, t, :]
            delta_t = delta[:, t]

            h_new, prog_prob = cell(
                x_t=x_t,
                x_prev=prev_x,
                h_prev=h,
                delta_t=delta_t,
            )

            h = torch.where(active.unsqueeze(1), h_new, h)

            if not reverse:
                is_last = lengths == (t + 1)
                last_valid_h = torch.where(is_last.unsqueeze(1), h, last_valid_h)

            prev_x = x_t

        if reverse:
            return h

        return last_valid_h

    def forward(self, x, lengths, delta=None, img_seq=None):
        if delta is None:
            raise ValueError("PSALMGRUDEncoder 需要 delta。")

        h_fw = self._run_one_direction(
            x=x,
            lengths=lengths,
            delta=delta,
            reverse=False,
        )

        if self.bidirectional:
            h_bw = self._run_one_direction(
                x=x,
                lengths=lengths,
                delta=delta,
                reverse=True,
            )
            return torch.cat([h_fw, h_bw], dim=1)

        return h_fw


class SimpleFusionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.output_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, lengths, delta=None, img_seq=None):
        B, T, D = x.shape
        device = x.device

        if torch.any(lengths <= 0):
            raise ValueError("SimpleFusionEncoder expects every sequence to have at least one visit.")

        time_index = torch.arange(T, device=device).unsqueeze(0)
        mask = time_index < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        mean_feat = (x * mask_f).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()

        neg_inf = -1.0e4
        max_input = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        max_feat = max_input.max(dim=1).values

        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(B, device=device)
        last_feat = x[batch_idx, last_idx, :]

        fused = torch.cat([mean_feat, max_feat, last_feat], dim=1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        return self.proj(fused)


def build_temporal_encoder(
    temporal_backbone,
    input_dim,
    hidden_dim=128,
    num_layers=1,
    dropout=0.3,
    bidirectional=True,
    img_dim=None,
):
    name = temporal_backbone.lower()

    if name in ["simple_fusion", "simple-fusion", "simple"]:
        return SimpleFusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if name == "gru":
        return GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    if name in ["ca_grud", "ca-grud", "cagrud"]:
        return CAGRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            img_dim=img_dim,
        )

    if name in ["psalm_grud", "psalm-grud", "psalm", "psa_grud"]:
        return PSALMGRUDEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            img_dim=img_dim,
        )

    raise ValueError(f"Unknown TEMPORAL_BACKBONE: {temporal_backbone}")
