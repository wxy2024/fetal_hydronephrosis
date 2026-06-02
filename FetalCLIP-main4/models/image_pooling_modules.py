from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMILPooling(nn.Module):
    """
    Gated Attention MIL Pooling.

    输入：
        image_features: [N, D]

    输出：
        pooled:     [D]
        image_mask: [1]
        attn:       [N]
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_dim = feature_dim

        self.norm = nn.LayerNorm(feature_dim)

        self.V = nn.Linear(feature_dim, hidden_dim)
        self.U = nn.Linear(feature_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        image_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        image_features = image_features.float().to(device)

        x = self.norm(image_features)

        h_v = torch.tanh(self.V(x))
        h_u = torch.sigmoid(self.U(x))
        h = h_v * h_u

        scores = self.w(h).squeeze(-1)

        attn = F.softmax(scores, dim=0)

        pooled = torch.sum(
            image_features * attn.unsqueeze(-1),
            dim=0,
        )

        image_mask = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )

        return pooled, image_mask, attn


class AttentionImagePooling(nn.Module):
    """
    普通 Attention Pooling。

    输入：
        image_features: [N, D]

    输出：
        pooled:     [D]
        image_mask: [1]
        attn:       [N]
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

        image_features = image_features.float().to(device)

        scores = self.score_net(image_features).squeeze(-1)

        attn = F.softmax(scores, dim=0)

        pooled = torch.sum(
            image_features * attn.unsqueeze(-1),
            dim=0,
        )

        image_mask = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )

        return pooled, image_mask, attn


class MeanImagePooling(nn.Module):
    """
    平均池化 baseline。
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_dim = feature_dim

    def forward(
        self,
        image_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        image_features = image_features.float().to(device)

        n = image_features.shape[0]

        pooled = image_features.mean(dim=0)

        image_mask = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )

        attn = torch.ones(
            n,
            dtype=torch.float32,
            device=device,
        ) / max(n, 1)

        return pooled, image_mask, attn


def build_image_pooling(
    pooling_type: str,
    feature_dim: int,
    hidden_dim: int = 256,
):
    """
    图像聚合模块工厂函数。

    pooling_type 可选：
        mean
        attention
        gated_mil
    """

    pooling_type = pooling_type.lower()

    if pooling_type == "mean":
        return MeanImagePooling(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    if pooling_type in ["attention", "attn"]:
        return AttentionImagePooling(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    if pooling_type in ["gated_mil", "gated_attention_mil", "mil"]:
        return GatedAttentionMILPooling(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    raise ValueError(f"Unknown image pooling type: {pooling_type}")