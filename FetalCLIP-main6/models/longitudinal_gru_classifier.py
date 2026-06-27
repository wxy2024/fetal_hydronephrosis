from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.image_pooling_modules import build_image_pooling
from models.visit_fusion_modules import build_visit_fusion
from models.temporal_encoders import build_temporal_encoder

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


class ModularGRUSideClassifier(nn.Module):
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
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        temporal_backbone: str = "gru",
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



        self.temporal_backbone = temporal_backbone

        self.temporal_encoder = build_temporal_encoder(
            temporal_backbone=temporal_backbone,
            input_dim=self.visit_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            img_dim=feature_dim,
        )

        gru_out_dim = self.temporal_encoder.output_dim




        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

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
        img_features_for_temporal = []
        interval_features = []
        phase_features = []
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
            img_features_for_temporal.append(img_feat)
            interval_features.append(interval_days / 100.0)
            phase_features.append(phase_flag)

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
        img_seq = torch.stack(img_features_for_temporal, dim=0)
        delta_seq = torch.tensor(
            interval_features,
            dtype=torch.float32,
            device=device,
        )
        phase_seq = torch.tensor(
            phase_features,
            dtype=torch.float32,
            device=device,
        )

        return seq, delta_seq, img_seq, phase_seq, attention_records

    def forward(
        self,
        batch: List[Dict],
        return_attention: bool = False,
    ):
        device = next(self.parameters()).device

        sequences = []
        delta_sequences = []
        img_sequences = []
        phase_sequences = []
        attention_batch = []

        for sample in batch:
            seq, delta_seq, img_seq, phase_seq, attn_records = self._build_one_sequence(
                sample,
                return_attention=return_attention,
            )
            sequences.append(seq)
            delta_sequences.append(delta_seq)
            img_sequences.append(img_seq)
            phase_sequences.append(phase_seq)
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

        padded_delta = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.float32,
            device=device,
        )

        padded_img = torch.zeros(
            batch_size,
            max_len,
            self.feature_dim,
            dtype=torch.float32,
            device=device,
        )

        padded_phase = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.float32,
            device=device,
        )

        for i, seq in enumerate(sequences):
            L = seq.shape[0]
            padded[i, :L, :] = seq
            padded_delta[i, :L] = delta_sequences[i]
            padded_img[i, :L, :] = img_sequences[i]
            padded_phase[i, :L] = phase_sequences[i]

        h_last = self.temporal_encoder(
            padded,
            lengths=lengths,
            delta=padded_delta,
            img_seq=padded_img,
            phase_seq=padded_phase,
        )

        logits = self.classifier(h_last).squeeze(-1)

        if return_attention:
            return logits, attention_batch

        return logits, None
