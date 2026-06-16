from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.image_pooling_modules import build_image_pooling
from models.visit_fusion_modules import build_visit_fusion

from physiopro.network.contiformer import ContiFormer as PhysioContiFormer


def parse_exam_date(visit: Dict) -> Optional[datetime]:
    exam_date = str(visit.get("exam_date", "")).strip()

    for fmt in ["%Y-%m-%d", "%Y%m%d"]:
        try:
            if exam_date:
                return datetime.strptime(exam_date, fmt)
        except Exception:
            pass

    visit_id = str(visit.get("visit_id", ""))

    if len(visit_id) >= 8 and visit_id[:8].isdigit():
        try:
            return datetime.strptime(visit_id[:8], "%Y%m%d")
        except Exception:
            return None

    return None


class ModularContiFormerSideClassifier(nn.Module):
    """
    模块化版本：

        image_features
        → image_pooling
        → img_feat

        img_feat + text_feat + struct_feat
        → visit_fusion
        → visit_feat

        visit_feat sequence
        → GRU
        → classifier
    """

    def __init__(
        self,
        feature_dim: int,
        image_pooling_type: str = "gated_mil",
        visit_fusion_type: str = "gated_multimodal",
        attn_hidden_dim: int = 256,
        fusion_hidden_dim: int = None,
        contiformer_d_model: int = 128,
        contiformer_d_inner: int = 256,
        contiformer_n_layers: int = 1,
        contiformer_n_head: int = 4,
        contiformer_d_k: int = 32,
        contiformer_d_v: int = 32,
        contiformer_method_ode: str = "rk4",
        contiformer_atol_ode: float = 1e-3,
        contiformer_rtol_ode: float = 1e-3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.image_pool = build_image_pooling(
            pooling_type=image_pooling_type,
            feature_dim=feature_dim,
            hidden_dim=attn_hidden_dim,
        )

        # struct_feat = image_mask + extra_feat
        # image_mask: 1
        # phase_flag: 1
        # age_days: 1
        # age_type: 1
        # days_since_first: 1
        # interval_days: 1
        self.struct_dim = 6

        self.visit_fusion = build_visit_fusion(
            fusion_type=visit_fusion_type,
            feature_dim=feature_dim,
            struct_dim=self.struct_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

        self.visit_dim = self.visit_fusion.output_dim

        # self.bidirectional = bidirectional
        # self.num_directions = 2 if bidirectional else 1

        # self.gru = nn.GRU(
        #     input_size=self.visit_dim,
        #     hidden_size=gru_hidden_dim,
        #     num_layers=gru_num_layers,
        #     batch_first=True,
        #     dropout=dropout if gru_num_layers > 1 else 0.0,
        #     bidirectional=bidirectional,
        # )

        # gru_out_dim = gru_hidden_dim * self.num_directions

        self.temporal_encoder = PhysioContiFormer(
            input_size=self.visit_dim,
            d_model=contiformer_d_model,
            d_inner=contiformer_d_inner,
            n_layers=contiformer_n_layers,
            n_head=contiformer_n_head,
            d_k=contiformer_d_k,
            d_v=contiformer_d_v,
            dropout=dropout,
            method_ode=contiformer_method_ode,
            atol_ode=contiformer_atol_ode,
            rtol_ode=contiformer_rtol_ode,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(contiformer_d_model),
            nn.Dropout(dropout),
            nn.Linear(contiformer_d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )



        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(gru_out_dim),
        #     nn.Dropout(dropout),
        #     nn.Linear(gru_out_dim, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(64, 1),
        # )

    def _tensor_to_device(
        self,
        value: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None

        device = next(self.parameters()).device
        return value.float().to(device)

    def _build_one_sequence(
        self,
        sample: Dict,
        return_attention: bool = False,
    ):
        device = next(self.parameters()).device

        visits = sample["visits"]

        dates = [parse_exam_date(v) for v in visits]
        valid_dates = [d for d in dates if d is not None]

        first_date = valid_dates[0] if len(valid_dates) > 0 else None
        prev_date = None

        visit_features = []
        visit_times = []

        attention_records = []

        for visit, exam_date in zip(visits, dates):
            image_features = self._tensor_to_device(
                visit.get("image_features", None)
            )

            text_feat = self._tensor_to_device(
                visit.get("text_feature", None)
            )

            if text_feat is None:
                text_feat = torch.zeros(
                    self.feature_dim,
                    dtype=torch.float32,
                    device=device,
                )

            #img_feat, image_mask, image_attn = self.image_pool(image_features)
            img_feat, image_mask, image_attn = self.image_pool(image_features,text_feat)

            phase_flag = float(visit.get("phase_flag", 0.0))

            if int(phase_flag) == 0:
                age_days = float(visit.get("gestational_age_days", 0.0))
                age_type = 0.0
            else:
                age_days = float(visit.get("postnatal_age_days", 0.0))
                age_type = 1.0

            if first_date is not None and exam_date is not None:
                days_since_first = float((exam_date - first_date).days)
            else:
                days_since_first = 0.0

            visit_times.append(days_since_first / 1000.0)

            if prev_date is not None and exam_date is not None:
                interval_days = float((exam_date - prev_date).days)
            else:
                interval_days = 0.0

            if exam_date is not None:
                prev_date = exam_date

            extra_feat = torch.tensor(
                [
                    phase_flag,
                    age_days / 300.0,
                    age_type,
                    days_since_first / 1000.0,
                    interval_days / 100.0,
                ],
                dtype=torch.float32,
                device=device,
            )

            struct_feat = torch.cat(
                [
                    image_mask,
                    extra_feat,
                ],
                dim=0,
            )  # [6]

            visit_feat, fusion_info = self.visit_fusion(
                img_feat=img_feat,
                text_feat=text_feat,
                struct_feat=struct_feat,
            )

            visit_features.append(visit_feat)

            if return_attention:
                image_names = [
                    Path(p).name
                    for p in visit.get("image_paths", [])
                ]

                record = {
                    "visit_id": str(visit.get("visit_id", "")),
                    "phase": str(visit.get("phase", "")),
                    "num_images": int(visit.get("num_images", 0)),
                    "image_names": image_names,
                    "image_attention": [
                        float(x)
                        for x in image_attn.detach().cpu().tolist()
                    ],
                }

                record.update(fusion_info)

                attention_records.append(record)

        seq = torch.stack(visit_features, dim=0)


        times = torch.tensor(visit_times, dtype=torch.float32, device=device)
        # 防止同一天检查或时间不递增导致 ODE 部分不稳定
        for i in range(1, times.shape[0]):
            if times[i] <= times[i - 1]:
                times[i] = times[i - 1] + 1e-4

        return seq, times, attention_records




    def forward(
        self,
        batch: List[Dict],
        return_attention: bool = False,
    ):
        device = next(self.parameters()).device

        sequences = []
        time_sequences = []
        attention_batch = []

        for sample in batch:
            seq, times, attn_records = self._build_one_sequence(
                sample,
                return_attention=return_attention,
            )
            sequences.append(seq)
            time_sequences.append(times)
            attention_batch.append(attn_records)

        lengths = torch.tensor(
            [seq.shape[0] for seq in sequences],
            dtype=torch.long,
            device=device,
        )

        batch_size = len(sequences)
        max_len = int(lengths.max().item())

        padded = torch.zeros(
            batch_size,
            max_len,
            self.visit_dim,
            dtype=torch.float32,
            device=device,
        )

        padded_times = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.float32,
            device=device,
        )

        for i, (seq, times) in enumerate(zip(sequences, time_sequences)):
            L = seq.shape[0]
            padded[i, :L, :] = seq
            padded_times[i, :L] = times

            if L < max_len:
                # padding 时间要略大于最后一个真实时间，避免重复时间
                extra = torch.arange(1, max_len - L + 1, device=device).float() * 1e-4
                padded_times[i, L:] = times[-1] + extra

        # packed = pack_padded_sequence(
        #     padded,
        #     lengths.cpu(),
        #     batch_first=True,
        #     enforce_sorted=False,
        # )

        # _, h_n = self.gru(packed)

        # if self.bidirectional:
        #     h_forward = h_n[-2]
        #     h_backward = h_n[-1]
        #     h_last = torch.cat(
        #         [
        #             h_forward,
        #             h_backward,
        #         ],
        #         dim=1,
        #     )
        # else:
        #     h_last = h_n[-1]







        # padded: [B, T, visit_dim]
        # padded_times: [B, T]
        # lengths: [B]

        invalid = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]

        # 先用这个 mask；如果 PhysioPro 版本因 mask 维度报错，
        # 可以临时改成 mask=None 先跑通。
        attn_mask = invalid[:, None, :]  # [B, 1, T]

        enc_out, _ = self.temporal_encoder(
            padded,
            t=padded_times,
            mask=attn_mask,
        )

        last_index = (lengths - 1).view(batch_size, 1, 1)
        last_index = last_index.expand(-1, 1, enc_out.shape[-1])
        h_last = enc_out.gather(dim=1, index=last_index).squeeze(1)








        logits = self.classifier(h_last).squeeze(-1)

        if return_attention:
            return logits, attention_batch

        return logits, None