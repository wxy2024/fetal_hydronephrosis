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





class TransformerTokenVisitFusion(nn.Module):
    """
    Transformer Token Fusion 检查级融合模块。

    输入：
        img_feat:    [D]
        text_feat:   [D]
        struct_feat: [S]

    构造 token：
        [CLS] [IMG] [TEXT] [STRUCT]

    输出：
        visit_feat: [D]

    其中：
        IMG token    来自图像特征
        TEXT token   来自文本特征
        STRUCT token 来自时间、阶段、image_mask 等结构化信息
        CLS token    用于汇总本次检查的多模态信息
    """

    def __init__(
        self,
        feature_dim: int,
        struct_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.0,
        num_heads: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.struct_dim = struct_dim

        if hidden_dim is None:
            hidden_dim = feature_dim * 4

        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim={feature_dim} 必须能被 num_heads={num_heads} 整除。"
                f"请把 num_heads 改成 1、2、4、8 中能整除 feature_dim 的数。"
            )

        self.img_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.struct_proj = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 一个可学习的 CLS token，用于汇总 IMG/TEXT/STRUCT 信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # 模态类型 embedding：CLS、IMG、TEXT、STRUCT 共 4 个 token
        self.type_embedding = nn.Parameter(torch.zeros(1, 4, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.out_norm = nn.LayerNorm(feature_dim)

        # 输出就是 CLS token 的维度
        self.output_dim = feature_dim

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
        struct_feat: torch.Tensor,
    ):
        """
        img_feat:    [D]
        text_feat:   [D]
        struct_feat: [S]
        """
        # 1. 各模态投影成同一维度 D
        img_token = self.img_proj(img_feat).unsqueeze(0)       # [1, D]
        text_token = self.text_proj(text_feat).unsqueeze(0)    # [1, D]
        struct_token = self.struct_proj(struct_feat).unsqueeze(0)  # [1, D]

        # 2. 拼成 token 序列：[CLS, IMG, TEXT, STRUCT]
        cls_token = self.cls_token.squeeze(0)  # [1, D]

        tokens = torch.cat(
            [
                cls_token,
                img_token,
                text_token,
                struct_token,
            ],
            dim=0,
        )  # [4, D]

        tokens = tokens.unsqueeze(0)  # [1, 4, D]

        # 3. 加模态类型 embedding
        tokens = tokens + self.type_embedding  # [1, 4, D]

        # 4. Transformer Encoder 融合
        fused_tokens = self.transformer(tokens)  # [1, 4, D]

        # 5. 取 CLS token 作为本次检查的融合表征
        visit_feat = fused_tokens[:, 0, :].squeeze(0)  # [D]

        visit_feat = self.out_norm(visit_feat)

        info = {
            "fusion_type": "transformer_token",
        }

        return visit_feat, info




class FiLMGatedMultimodalVisitFusion(nn.Module):
    """
    图像-文本门控融合 + 时间/阶段条件调制 FiLM。

    流程：
        img_feat + text_feat
        → Gated Multimodal Fusion
        → img_text_feat

        struct_feat
        → gamma, beta

        visit_feat = gamma * img_text_feat + beta

    这里 struct_feat 包括：
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

        # 图像投影
        self.img_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 文本投影
        self.text_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 图文门控融合
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # 将图文融合后的 hidden_dim 映射回 feature_dim
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )

        # FiLM 条件调制分支
        # 输入 struct_feat，输出 gamma 和 beta
        self.film_net = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

        self.final_norm = nn.LayerNorm(feature_dim)

        # FiLM 后不再拼接 struct_feat，所以输出维度就是 feature_dim
        self.output_dim = feature_dim

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
        struct_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        # 1. 图像和文本分别投影
        img_h = self.img_proj(img_feat)       # [H]
        text_h = self.text_proj(text_feat)    # [H]

        # 2. 图文门控融合
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

        # 3. struct_feat 生成 FiLM 参数
        film_params = self.film_net(struct_feat)  # [2D]

        gamma, beta = torch.chunk(
            film_params,
            chunks=2,
            dim=-1,
        )  # gamma: [D], beta: [D]

        # 这里建议让 gamma 以 1 为中心，训练更稳定
        gamma = 1.0 + torch.tanh(gamma)

        # beta 用 tanh 限制幅度，避免初期过度偏移
        beta = torch.tanh(beta)

        # 4. FiLM 调制
        visit_feat = gamma * img_text_feat + beta

        visit_feat = self.final_norm(visit_feat)

        gate_mean = float(gate.detach().mean().cpu().item())
        gamma_mean = float(gamma.detach().mean().cpu().item())
        beta_mean = float(beta.detach().mean().cpu().item())

        info = {
            "fusion_type": "film_gated_multimodal",
            "image_text_gate_mean": gate_mean,
            "image_weight_mean": gate_mean,
            "text_weight_mean": 1.0 - gate_mean,
            "film_gamma_mean": gamma_mean,
            "film_beta_mean": beta_mean,
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

    if fusion_type in ["film", "film_gated_multimodal", "film_gmf"]:
        return FiLMGatedMultimodalVisitFusion(
            feature_dim=feature_dim,
            struct_dim=struct_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


    if fusion_type in ["transformer_token", "token_transformer", "ttf"]:
        return TransformerTokenVisitFusion(
            feature_dim=feature_dim,
            struct_dim=struct_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=4,
            num_layers=1,
        )

    raise ValueError(f"Unknown visit fusion type: {fusion_type}")