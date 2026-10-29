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

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
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


class GRUDCell(nn.Module):
    """
    GRU-D cell for irregularly sampled visit sequences.

    Since the visit-level feature is already fused and has no explicit missing
    mask, this implementation uses inter-visit interval to decay both the input
    feature and hidden state before a GRUCell update.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.x_mean = nn.Parameter(torch.zeros(input_dim))
        self.input_decay = nn.Linear(1, input_dim)
        self.hidden_decay = nn.Linear(1, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _gamma(self, layer, delta_t):
        return torch.exp(-torch.relu(layer(delta_t.unsqueeze(-1))))

    def forward(self, x_t, h_prev, delta_t):
        delta_t = torch.clamp(delta_t, min=0.0)

        gamma_x = self._gamma(self.input_decay, delta_t)
        gamma_h = self._gamma(self.hidden_decay, delta_t)

        x_hat = gamma_x * x_t + (1.0 - gamma_x) * self.x_mean.unsqueeze(0)
        h_hat = gamma_h * h_prev

        h_t = self.gru_cell(x_hat, h_hat)
        h_t = self.dropout(h_t)
        return h_t


class GRUDEncoder(nn.Module):
    """
    GRU-D encoder with optional bidirectional processing.

    TEMPORAL_BACKBONE values: gru_d / grud.
    """

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
            raise ValueError("GRUDEncoder currently supports num_layers=1.")

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.fw_cell = GRUDCell(input_dim, hidden_dim, dropout)
        if bidirectional:
            self.bw_cell = GRUDCell(input_dim, hidden_dim, dropout)

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def _direction_delta(self, delta, t, reverse, max_len):
        if not reverse:
            return delta[:, t]
        if t + 1 < max_len:
            return delta[:, t + 1]
        return torch.zeros_like(delta[:, t])

    def _run_one_direction(self, x, lengths, delta, reverse=False):
        B, T, _ = x.shape
        device = x.device

        cell = self.bw_cell if reverse else self.fw_cell
        h = torch.zeros(B, self.hidden_dim, device=device)
        last_valid_h = h
        time_range = range(T - 1, -1, -1) if reverse else range(T)

        for t in time_range:
            active = lengths > t
            if not active.any():
                continue

            x_t = x[:, t, :]
            delta_t = self._direction_delta(delta, t, reverse, T)
            h_new = cell(x_t=x_t, h_prev=h, delta_t=delta_t)
            h = torch.where(active.unsqueeze(1), h_new, h)

            if not reverse:
                is_last = lengths == (t + 1)
                last_valid_h = torch.where(is_last.unsqueeze(1), h, last_valid_h)

        if reverse:
            return h

        return last_valid_h

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
        if delta is None:
            raise ValueError("GRUDEncoder requires delta.")

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

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
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

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
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


class BiTGRUCell(nn.Module):
    """
    Bidirectional Time-aware GRU cell.

    The cell uses irregular inter-visit interval, prenatal/postnatal phase, and
    adjacent representation change to control hidden-state decay and update.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_mean = nn.Parameter(torch.zeros(input_dim))

        self.input_decay = nn.Linear(2, input_dim)
        self.hidden_decay = nn.Linear(2, hidden_dim)

        self.update_gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.phase_gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _gamma(self, layer, decay_feat):
        return torch.exp(-torch.relu(layer(decay_feat)))

    def forward(self, x_t, x_prev, h_prev, delta_t, phase_t):
        delta_t = torch.clamp(delta_t, min=0.0)
        phase_t = torch.clamp(phase_t, min=0.0, max=1.0)

        change = torch.linalg.vector_norm(
            x_t - x_prev,
            ord=2,
            dim=-1,
        ) / (x_t.shape[-1] ** 0.5)
        change = torch.clamp(change, min=0.0, max=10.0)

        decay_feat = torch.stack([delta_t, phase_t], dim=-1)
        gamma_x = self._gamma(self.input_decay, decay_feat)
        gamma_h = self._gamma(self.hidden_decay, decay_feat)

        x_hat = gamma_x * x_t + (1.0 - gamma_x) * self.x_mean.unsqueeze(0)
        h_decay = gamma_h * h_prev

        h_candidate = self.gru_cell(x_hat, h_decay)
        h_candidate = self.dropout(h_candidate)

        update_feat = torch.stack([delta_t, phase_t, change], dim=-1)
        update_gate = self.update_gate(update_feat)

        phase_feat = torch.stack([phase_t, delta_t], dim=-1)
        phase_gate = self.phase_gate(phase_feat)

        h_update = update_gate * h_candidate + (1.0 - update_gate) * h_decay
        h_t = phase_gate * h_update + (1.0 - phase_gate) * h_prev
        return h_t


class BiTGRUEncoder(nn.Module):
    """
    BiT-GRU: Bidirectional Time-aware Gated Residual Unit.

    It keeps the strong bidirectional GRU inductive bias, but explicitly models
    irregular sampling intervals and prenatal/postnatal phase transitions.
    """

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
            raise ValueError("BiTGRUEncoder currently supports num_layers=1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.fw_cell = BiTGRUCell(input_dim, hidden_dim, dropout)
        if bidirectional:
            self.bw_cell = BiTGRUCell(input_dim, hidden_dim, dropout)

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def _direction_delta(self, delta, t, reverse, max_len):
        if not reverse:
            return delta[:, t]
        if t + 1 < max_len:
            return delta[:, t + 1]
        return torch.zeros_like(delta[:, t])

    def _run_one_direction(self, x, lengths, delta, phase_seq, reverse=False):
        B, T, _ = x.shape
        device = x.device

        cell = self.bw_cell if reverse else self.fw_cell
        h = torch.zeros(B, self.hidden_dim, device=device)
        last_valid_h = h

        time_range = range(T - 1, -1, -1) if reverse else range(T)
        prev_x = x[:, T - 1, :] if reverse else x[:, 0, :]

        for t in time_range:
            active = lengths > t
            if not active.any():
                continue

            x_t = x[:, t, :]
            delta_t = self._direction_delta(delta, t, reverse, T)
            phase_t = phase_seq[:, t]

            h_new = cell(
                x_t=x_t,
                x_prev=prev_x,
                h_prev=h,
                delta_t=delta_t,
                phase_t=phase_t,
            )
            h = torch.where(active.unsqueeze(1), h_new, h)

            if not reverse:
                is_last = lengths == (t + 1)
                last_valid_h = torch.where(is_last.unsqueeze(1), h, last_valid_h)

            prev_x = x_t

        if reverse:
            return h

        return last_valid_h

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
        if delta is None:
            raise ValueError("BiTGRUEncoder requires delta.")

        if phase_seq is None:
            phase_seq = torch.zeros(
                x.shape[:2],
                dtype=torch.float32,
                device=x.device,
            )

        h_fw = self._run_one_direction(
            x=x,
            lengths=lengths,
            delta=delta,
            phase_seq=phase_seq,
            reverse=False,
        )

        if self.bidirectional:
            h_bw = self._run_one_direction(
                x=x,
                lengths=lengths,
                delta=delta,
                phase_seq=phase_seq,
                reverse=True,
            )
            return torch.cat([h_fw, h_bw], dim=1)

        return h_fw


class TriPGGRUCell(nn.Module):
    """
    Tri-state Progression-Gated GRU cell.

    The cell explicitly compares the current visit with the previous visit,
    estimates latent progression states, and uses irregular time intervals to
    control hidden-state decay and update.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=0.3,
        progression_dim=None,
        eps=1e-6,
    ):
        super().__init__()

        if progression_dim is None:
            progression_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        state_in_dim = input_dim * 4 + 3
        delta_in_dim = input_dim * 2 + 3

        self.progression_state_net = nn.Sequential(
            nn.LayerNorm(state_in_dim),
            nn.Linear(state_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self.progression_embed = nn.Sequential(
            nn.LayerNorm(delta_in_dim),
            nn.Linear(delta_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, progression_dim),
        )

        self.embed_to_input = nn.Sequential(
            nn.LayerNorm(progression_dim),
            nn.Linear(progression_dim, input_dim),
            nn.GELU(),
        )

        self.inject_gate = nn.Sequential(
            nn.Linear(5, input_dim),
            nn.Sigmoid(),
        )

        self.hidden_decay = nn.Linear(4, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _gamma(self, feat):
        return torch.exp(-torch.relu(self.hidden_decay(feat)))

    def forward(self, x_t, x_prev, h_prev, delta_t, phase_t):
        delta_t = torch.clamp(delta_t, min=0.0)
        phase_t = torch.clamp(phase_t, min=0.0, max=1.0)

        diff = x_t - x_prev
        abs_diff = torch.abs(diff)
        rate = abs_diff / (delta_t.unsqueeze(-1) + 1.0 + self.eps)

        diff_norm = torch.linalg.vector_norm(
            diff,
            ord=2,
            dim=-1,
        ) / (x_t.shape[-1] ** 0.5)
        diff_norm = torch.clamp(diff_norm, min=0.0, max=10.0)

        rate_norm = torch.linalg.vector_norm(
            rate,
            ord=2,
            dim=-1,
        ) / (x_t.shape[-1] ** 0.5)
        rate_norm = torch.clamp(rate_norm, min=0.0, max=10.0)

        scalar_feat = torch.stack(
            [
                delta_t,
                phase_t,
                diff_norm,
            ],
            dim=-1,
        )

        state_input = torch.cat(
            [
                x_t,
                x_prev,
                diff,
                abs_diff,
                scalar_feat,
            ],
            dim=-1,
        )
        state_logits = self.progression_state_net(state_input)
        state_prob = F.softmax(state_logits, dim=-1)

        delta_input = torch.cat(
            [
                diff,
                abs_diff,
                torch.stack([delta_t, phase_t, rate_norm], dim=-1),
            ],
            dim=-1,
        )
        progression = self.embed_to_input(self.progression_embed(delta_input))

        stable_prob = state_prob[:, 0]
        improve_prob = state_prob[:, 1]
        worsen_prob = state_prob[:, 2]

        gate_input = torch.stack(
            [
                stable_prob,
                improve_prob,
                worsen_prob,
                delta_t,
                phase_t,
            ],
            dim=-1,
        )
        inject_gate = self.inject_gate(gate_input)
        x_aug = x_t + inject_gate * progression

        decay_input = torch.cat(
            [
                delta_t.unsqueeze(-1),
                state_prob,
            ],
            dim=-1,
        )
        gamma_h = self._gamma(decay_input)
        h_decay = gamma_h * h_prev

        h_candidate = self.gru_cell(x_aug, h_decay)
        h_candidate = self.dropout(h_candidate)

        write_strength = (1.0 - stable_prob).unsqueeze(-1)
        h_t = stable_prob.unsqueeze(-1) * h_decay + write_strength * h_candidate
        return h_t


class TriPGGRUEncoder(nn.Module):
    """
    TriPG-GRU: single-direction Tri-state Progression-Gated GRU.

    This encoder is intentionally forward-only because progression is defined as
    the change from the previous visit to the current visit.
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
            raise ValueError("TriPGGRUEncoder currently supports num_layers=1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.cell = TriPGGRUCell(input_dim, hidden_dim, dropout)

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
        if delta is None:
            raise ValueError("TriPGGRUEncoder requires delta.")

        if phase_seq is None:
            phase_seq = torch.zeros(
                x.shape[:2],
                dtype=torch.float32,
                device=x.device,
            )

        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        last_valid_h = h
        prev_x = x[:, 0, :]

        for t in range(T):
            active = lengths > t
            if not active.any():
                continue

            x_t = x[:, t, :]
            delta_t = delta[:, t]
            phase_t = phase_seq[:, t]

            h_new = self.cell(
                x_t=x_t,
                x_prev=prev_x,
                h_prev=h,
                delta_t=delta_t,
                phase_t=phase_t,
            )
            h = torch.where(active.unsqueeze(1), h_new, h)

            is_last = lengths == (t + 1)
            last_valid_h = torch.where(is_last.unsqueeze(1), h, last_valid_h)
            prev_x = torch.where(active.unsqueeze(1), x_t, prev_x)

        return last_valid_h


class LatentRiskGRUCell(nn.Module):
    """
    Latent Risk-Guided GRU cell.

    Each visit receives the previous latent risk score in addition to the visit
    representation and temporal context, then predicts the current risk score.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.LayerNorm(input_dim + 3),
            nn.Linear(input_dim + 3, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + input_dim + 3),
            nn.Linear(hidden_dim + input_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, h_prev, prev_score, delta_t, phase_t):
        delta_t = torch.clamp(delta_t, min=0.0)
        phase_t = torch.clamp(phase_t, min=0.0, max=1.0)
        prev_score = torch.clamp(prev_score, min=0.0, max=1.0)

        scalar_context = torch.stack(
            [
                prev_score,
                delta_t,
                phase_t,
            ],
            dim=-1,
        )
        x_aug = self.context_proj(torch.cat([x_t, scalar_context], dim=-1))
        h_t = self.gru_cell(x_aug, h_prev)

        risk_input = torch.cat([h_t, x_t, scalar_context], dim=-1)
        risk_logit = self.risk_head(risk_input).squeeze(-1)
        risk_score = torch.sigmoid(risk_logit)
        return h_t, risk_logit, risk_score


class LatentRiskGRUEncoder(nn.Module):
    """
    Latent Risk-Guided GRU.

    The encoder is forward-only. It initializes the previous visit risk as 0.5,
    recursively predicts a visit-level risk score, and returns the final visit
    risk logit directly for BCE supervision.
    """

    direct_logit = True

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
            raise ValueError("LatentRiskGRUEncoder currently supports num_layers=1.")

        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.cell = LatentRiskGRUCell(input_dim, hidden_dim, dropout)

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
        if delta is None:
            raise ValueError("LatentRiskGRUEncoder requires delta.")

        if phase_seq is None:
            phase_seq = torch.zeros(
                x.shape[:2],
                dtype=torch.float32,
                device=x.device,
            )

        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        prev_score = torch.full(
            (B,),
            0.5,
            dtype=torch.float32,
            device=device,
        )
        last_risk_logit = torch.zeros(B, dtype=torch.float32, device=device)

        for t in range(T):
            active = lengths > t
            if not active.any():
                continue

            h_new, risk_logit, risk_score = self.cell(
                x_t=x[:, t, :],
                h_prev=h,
                prev_score=prev_score,
                delta_t=delta[:, t],
                phase_t=phase_seq[:, t],
            )

            h = torch.where(active.unsqueeze(1), h_new, h)
            prev_score = torch.where(active, risk_score, prev_score)

            is_last = lengths == (t + 1)
            last_risk_logit = torch.where(is_last, risk_logit, last_risk_logit)

        return last_risk_logit.unsqueeze(-1)


class DepthwiseSeparableTemporalConv(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [B, T, D], mask: [B, T]
        x_masked = x * mask.unsqueeze(-1).float()
        y = self.depthwise(x_masked.transpose(1, 2))
        y = self.pointwise(y).transpose(1, 2)
        y = self.dropout(F.gelu(y))
        y = y * mask.unsqueeze(-1).float()
        return self.norm(x + y)


class TDConvGRUEncoder(nn.Module):
    """
    Temporal-Difference Convolutional GRU.

    A lightweight multi-scale temporal convolution first captures local
    visit-to-visit patterns under irregular sampling, then a BiGRU models the
    full longitudinal context.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
        img_dim=None,
        d_model=512,
        use_temporal_conv=True,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.time_phase_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.use_temporal_conv = use_temporal_conv
        if use_temporal_conv:
            self.short_conv = DepthwiseSeparableTemporalConv(
                d_model=d_model,
                kernel_size=3,
                dropout=dropout,
            )
            self.mid_conv = DepthwiseSeparableTemporalConv(
                d_model=d_model,
                kernel_size=5,
                dropout=dropout,
            )

        self.local_fusion = nn.Sequential(
            nn.LayerNorm(d_model * (3 if use_temporal_conv else 1)),
            nn.Linear(d_model * (3 if use_temporal_conv else 1), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
        B, T, _ = x.shape
        device = x.device

        if delta is None:
            delta = torch.zeros(B, T, dtype=torch.float32, device=device)

        if phase_seq is None:
            phase_seq = torch.zeros(B, T, dtype=torch.float32, device=device)

        time_index = torch.arange(T, device=device).unsqueeze(0)
        mask = time_index < lengths.unsqueeze(1)

        delta = torch.clamp(delta, min=0.0)
        phase_seq = torch.clamp(phase_seq, min=0.0, max=1.0)
        time_phase = torch.stack([delta, phase_seq], dim=-1)

        base = self.input_proj(x)
        conditioned = base + self.time_phase_proj(time_phase)
        conditioned = conditioned * mask.unsqueeze(-1).float()

        if self.use_temporal_conv:
            short_feat = self.short_conv(conditioned, mask)
            mid_feat = self.mid_conv(conditioned, mask)
            fusion_input = torch.cat([conditioned, short_feat, mid_feat], dim=-1)
        else:
            fusion_input = conditioned
        enhanced = self.local_fusion(fusion_input)
        enhanced = enhanced * mask.unsqueeze(-1).float()

        packed = pack_padded_sequence(
            enhanced,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)

        if self.gru.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)

        return h_n[-1]


class TDConvGRUNoConvEncoder(TDConvGRUEncoder):
    """MT-ConvGRU ablation without temporal convolution.

    Keeps input projection, explicit time/phase conditioning, pointwise
    fusion and GRU. Fusion takes one d_model branch rather than three;
    consequently parameter count is reduced. No convolution is instantiated.
    This is distinct from both ordinary GRU and no-explicit-time ablations.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3,
                 bidirectional=True, img_dim=None, d_model=512):
        super().__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=bidirectional, img_dim=img_dim,
            d_model=d_model, use_temporal_conv=False,
        )


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

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
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


class MeanPoolingEncoder(nn.Module):
    """对同一病例的所有有效检查级特征沿时间维度求平均。"""

    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, x, lengths, delta=None, img_seq=None, phase_seq=None):
        if torch.any(lengths <= 0):
            raise ValueError(
                "MeanPoolingEncoder expects every sequence to have at least one visit."
            )

        # x: [B, T, D]；只平均真实检查，忽略 batch 中用于补齐的时间步。
        lengths = lengths.to(x.device)
        time_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        valid_mask = time_index < lengths.unsqueeze(1)
        valid_mask = valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        summed = (x * valid_mask).sum(dim=1)
        mean_feat = summed / lengths.unsqueeze(1).to(dtype=x.dtype)
        return torch.nan_to_num(
            mean_feat,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        )


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

    if name in ["mean_pooling", "mean-pooling", "mean"]:
        return MeanPoolingEncoder(input_dim=input_dim)

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

    if name in ["gru_d", "gru-d", "grud"]:
        return GRUDEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            img_dim=img_dim,
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

    if name in ["bit_gru", "bit-gru", "bitgru", "time_aware_gru", "time-aware-gru"]:
        return BiTGRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            img_dim=img_dim,
        )

    if name in [
        "tripg_gru",
        "tripg-gru",
        "tripg",
        "progression_gru",
        "progression-gated-gru",
    ]:
        return TriPGGRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            img_dim=img_dim,
        )

    if name in [
        "risk_gru",
        "risk-gru",
        "lrg_gru",
        "lrg-gru",
        "latent_risk_gru",
    ]:
        return LatentRiskGRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            img_dim=img_dim,
        )

    if name == "td_conv_gru_no_conv":
        return TDConvGRUNoConvEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=bidirectional, img_dim=img_dim,
            d_model=512,
        )

    if name in [
        "td_conv_gru",
        "td-conv-gru",
        "tdconvgru",
        "temporal_difference_conv_gru",
    ]:
        return TDConvGRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            img_dim=img_dim,
            d_model=512,
        )

    raise ValueError(f"Unknown TEMPORAL_BACKBONE: {temporal_backbone}")
