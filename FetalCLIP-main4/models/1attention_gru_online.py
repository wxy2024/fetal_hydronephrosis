from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class ImageAttentionPooling(nn.Module):
    """
    替代原来的 mean pooling。

    输入：
        一次检查内多张图像的 FetalCLIP 特征 [N, D]

    输出：
        该检查的图像级聚合特征 [D]
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_dim = feature_dim

        self.score_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        image_features:
            [N, D]，N 是一次检查内的图像数量。

        返回：
            pooled: [D]
            image_mask: [1]
            attn: [N]
        """
        device = next(self.parameters()).device

        if image_features is None or image_features.numel() == 0:
            pooled = torch.zeros(
                self.feature_dim,
                dtype=torch.float32,
                device=device,
            )
            image_mask = torch.tensor(
                [0.0],
                dtype=torch.float32,
                device=device,
            )
            attn = torch.empty(
                0,
                dtype=torch.float32,
                device=device,
            )
            return pooled, image_mask, attn

        image_features = image_features.float().to(device)  # [N, D]

        scores = self.score_net(image_features).squeeze(-1)  # [N]
        attn = F.softmax(scores, dim=0)  # [N]

        pooled = torch.sum(
            image_features * attn.unsqueeze(-1),
            dim=0,
        )  # [D]

        image_mask = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )

        return pooled, image_mask, attn


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


class AttentionGRUSideClassifier(nn.Module):
    """
    FetalCLIP 冻结特征 + 图像注意力池化 + GRU 病例侧别级分类。

    每个 visit 的特征构成：

        [
            attention_pooled_img,
            text_feat,
            image_mask,
            phase,
            age,
            age_type,
            days_since_first,
            interval_days
        ]
    """

    def __init__(
        self,
        feature_dim: int,
        attn_hidden_dim: int = 256,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.image_pool = ImageAttentionPooling(
            feature_dim=feature_dim,
            hidden_dim=attn_hidden_dim,
        )

        # image_mask 1 维
        # phase 1 维
        # age 1 维
        # age_type 1 维
        # days_since_first 1 维
        # interval_days 1 维
        self.extra_dim = 6

        self.visit_dim = feature_dim * 2 + self.extra_dim

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=self.visit_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = gru_hidden_dim * self.num_directions

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

            img_feat, image_mask, attn = self.image_pool(image_features)

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

            # image_mask 是 [1]，extra_feat 是 [5]，总共 6 个结构化特征。
            visit_feat = torch.cat(
                [
                    img_feat,
                    text_feat,
                    image_mask,
                    extra_feat,
                ],
                dim=0,
            )

            visit_features.append(visit_feat)

            if return_attention:
                image_names = [
                    Path(p).name
                    for p in visit.get("image_paths", [])
                ]

                attention_records.append(
                    {
                        "visit_id": str(visit.get("visit_id", "")),
                        "phase": str(visit.get("phase", "")),
                        "num_images": int(visit.get("num_images", 0)),
                        "image_names": image_names,
                        "attention": [
                            float(x)
                            for x in attn.detach().cpu().tolist()
                        ],
                    }
                )

        seq = torch.stack(visit_features, dim=0)  # [T, D_visit]

        return seq, attention_records

    def forward(
        self,
        batch: List[Dict],
        return_attention: bool = False,
    ):
        device = next(self.parameters()).device

        sequences = []
        attention_batch = []

        for sample in batch:
            seq, attn_records = self._build_one_sequence(
                sample,
                return_attention=return_attention,
            )
            sequences.append(seq)
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

        for i, seq in enumerate(sequences):
            padded[i, : seq.shape[0], :] = seq

        packed = pack_padded_sequence(
            padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, h_n = self.gru(packed)

        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_last = h_n[-1]

        logits = self.classifier(h_last).squeeze(-1)

        if return_attention:
            return logits, attention_batch

        return logits, None