import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3, bidirectional=True):
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
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
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
        # delta_t: [B]
        # 图像变化越大，说明前一次状态越不可信
        img_t_h = self.img_proj(img_t)
        img_prev_h = self.img_proj(img_prev)

        cos_sim = torch.cosine_similarity(img_t_h, img_prev_h, dim=-1)
        img_change = 1.0 - cos_sim
        img_change = torch.clamp(img_change, min=0.0, max=2.0)

        decay_feat = torch.stack([delta_t, img_change], dim=-1)  # [B, 2]

        gamma_x = self._gamma(self.input_decay, decay_feat)
        gamma_h = self._gamma(self.hidden_decay, decay_feat)

        x_hat = gamma_x * x_t + (1.0 - gamma_x) * self.x_mean.unsqueeze(0)
        h_hat = gamma_h * h_prev

        h_t = self.gru_cell(x_hat, h_hat)
        h_t = self.dropout(h_t)
        return h_t


class CAGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3, bidirectional=True, img_dim=None):
        super().__init__()
        if num_layers != 1:
            raise ValueError("CAGRUEncoder 当前建议先用 num_layers=1，便于和论文创新点保持清晰。")
        if img_dim is None:
            img_dim = input_dim

        self.bidirectional = bidirectional
        self.fw_cell = CAGRUCell(input_dim, hidden_dim, img_dim, dropout)

        if bidirectional:
            self.bw_cell = CAGRUCell(input_dim, hidden_dim, img_dim, dropout)

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def _run_one_direction(self, x, lengths, delta, img_seq, reverse=False):
        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.fw_cell.hidden_dim, device=device)

        cell = self.bw_cell if reverse else self.fw_cell
        time_range = range(T - 1, -1, -1) if reverse else range(T)

        last_valid_h = h
        prev_img = img_seq[:, 0, :] if not reverse else img_seq[:, T - 1, :]

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

        h_fw = self._run_one_direction(x, lengths, delta, img_seq, reverse=False)

        if self.bidirectional:
            h_bw = self._run_one_direction(x, lengths, delta, img_seq, reverse=True)
            return torch.cat([h_fw, h_bw], dim=1)

        return h_fw


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

    raise ValueError(f"Unknown TEMPORAL_BACKBONE: {temporal_backbone}")