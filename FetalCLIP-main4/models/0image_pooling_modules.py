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

    # def forward(
    #     self,
    #     image_features: Optional[torch.Tensor],
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def forward(
            self,
            image_features: Optional[torch.Tensor],
            query_feat: Optional[torch.Tensor] = None,
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
        query_feat: Optional[torch.Tensor] = None,
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

    输入：
        image_features: [N, D]
        query_feat: [D]，可选。当前 Mean Pooling 不使用它。

    输出：
        pooled: [D]
        image_mask: [1]
        attn: [N]
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_dim = feature_dim

    def forward(
        self,
        image_features: Optional[torch.Tensor],
        query_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if image_features is not None and image_features.numel() > 0:
            device = image_features.device
        elif query_feat is not None:
            device = query_feat.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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




class TextGuidedCrossAttentionImagePooling(nn.Module):
    """
    文本引导的跨注意力图像聚合。

    Query: 文本特征 text_feat [D]
    Key:   多张图像特征 image_features [N, D]
    Value: 多张图像特征 image_features [N, D]

    输出：
        pooled:     [D]
        image_mask: [1]
        attn:       [N]
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim={feature_dim} 必须能被 num_heads={num_heads} 整除"
            )

        self.query_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        image_features: Optional[torch.Tensor],
        query_feat: Optional[torch.Tensor] = None,
    ):
        device = next(self.parameters()).device

        if image_features is None or image_features.numel() == 0:
            pooled = torch.zeros(self.feature_dim, dtype=torch.float32, device=device)
            image_mask = torch.tensor([0.0], dtype=torch.float32, device=device)
            attn = torch.empty(0, dtype=torch.float32, device=device)
            return pooled, image_mask, attn

        image_features = image_features.float().to(device)   # [N, D]

        if query_feat is None:
            # 没有文本时，给一个 0 向量，保证代码不炸
            query_feat = torch.zeros(
                self.feature_dim,
                dtype=torch.float32,
                device=device,
            )
        else:
            query_feat = query_feat.float().to(device)       # [D]

        query = self.query_proj(query_feat).unsqueeze(0).unsqueeze(0)   # [1, 1, D]
        kv = self.kv_proj(image_features).unsqueeze(0)                  # [1, N, D]

        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=True,
        )

        pooled = attn_out.squeeze(0).squeeze(0)  # [D]
        pooled = self.out_norm(pooled + self.ffn(pooled))

        attn = attn_weights.squeeze(0).squeeze(0)  # [N]

        image_mask = torch.tensor([1.0], dtype=torch.float32, device=device)

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
    
    
    if pooling_type in ["text_cross_attention", "text_cross_attn", "tca"]:
        return TextGuidedCrossAttentionImagePooling(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
        )

    raise ValueError(f"Unknown image pooling type: {pooling_type}")