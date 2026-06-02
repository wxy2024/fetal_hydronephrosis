## 这是一个专门负责融合  图像特征+文本特征+时间等其他特征   的模块，提供不同的融合策略实现。

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DirectConcatVisitFusion(nn.Module):
    """
    原始版本：直接拼接。

    输入：
        img_feat:    [D]
        text_feat:   [D]
        struct_feat: [S]，例如 image_mask + phase + age + interval 等

    输出：
        visit_feat: [2D + S]
    """

    def __init__(
        self,
        feature_dim: int,
        struct_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.struct_dim = struct_dim
        self.output_dim = feature_dim * 2 + struct_dim

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
        struct_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        visit_feat = torch.cat(
            [
                img_feat,
                text_feat,
                struct_feat,
            ],
            dim=0,
        )

        info = {
            "fusion_type": "direct_concat",
        }

        return visit_feat, info


class GatedMultimodalVisitFusion(nn.Module):
    """
    图像-文本门控多模态融合 + 时间/阶段信息拼接。

    流程：
        img_feat + text_feat
        → Gated Multimodal Fusion
        → img_text_feat

        img_text_feat + struct_feat
        → visit_feat

    其中 struct_feat 可以包括：
        image_mask
        phase_flag
        age_days
        age_type
        days_since_first
        interval_days
    """

    def __init__(
        self,
        feature_dim: int,
        struct_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = feature_dim

        self.feature_dim = feature_dim
        self.struct_dim = struct_dim
        self.hidden_dim = hidden_dim

        self.img_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )

        # 图文融合后是 D 维，再拼接 struct_feat
        self.output_dim = feature_dim + struct_dim

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
        struct_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        img_h = self.img_proj(img_feat)       # [H]
        text_h = self.text_proj(text_feat)    # [H]

        fusion_input = torch.cat(
            [
                img_h,
                text_h,
            ],
            dim=-1,
        )  # [2H]

        gate = self.gate_net(fusion_input)  # [H]

        fused_h = gate * img_h + (1.0 - gate) * text_h  # [H]

        img_text_feat = self.out_proj(fused_h)  # [D]

        visit_feat = torch.cat(
            [
                img_text_feat,
                struct_feat,
            ],
            dim=0,
        )  # [D + S]

        gate_mean = float(gate.detach().mean().cpu().item())

        info = {
            "fusion_type": "gated_multimodal",
            "image_text_gate_mean": gate_mean,
            "image_weight_mean": gate_mean,
            "text_weight_mean": 1.0 - gate_mean,
        }

        return visit_feat, info


def build_visit_fusion(
    fusion_type: str,
    feature_dim: int,
    struct_dim: int,
    hidden_dim: int = None,
    dropout: float = 0.0,
):
    """
    检查级融合模块工厂函数。

    fusion_type 可选：
        direct_concat
        gated_multimodal
    """

    fusion_type = fusion_type.lower()

    if fusion_type == "direct_concat":
        return DirectConcatVisitFusion(
            feature_dim=feature_dim,
            struct_dim=struct_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if fusion_type in ["gated_multimodal", "gmf", "gated_img_text"]:
        return GatedMultimodalVisitFusion(
            feature_dim=feature_dim,
            struct_dim=struct_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    raise ValueError(f"Unknown visit fusion type: {fusion_type}")