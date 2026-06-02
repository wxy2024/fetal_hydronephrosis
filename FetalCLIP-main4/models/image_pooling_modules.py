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
        query_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        query_feat 是预留接口。
        当前 Gated MIL 不使用它。
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

        x = self.norm(image_features)  # [N, D]

        h_v = torch.tanh(self.V(x))       # [N, H]
        h_u = torch.sigmoid(self.U(x))    # [N, H]
        h = h_v * h_u                     # [N, H]

        scores = self.w(h).squeeze(-1)    # [N]

        attn = F.softmax(scores, dim=0)   # [N]

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


class SimilarityAwareRedundancySuppressedGatedAttentionPooling(nn.Module):
    """
    Similarity-aware Redundancy-suppressed Gated Attention Pooling.

    中文名：
        相似性感知的冗余抑制门控注意力池化模块。

    设计目的：
        针对单次超声检查中多张图像可能存在高度相似、
        重复测量或近重复切面的问题，在 Gated Attention MIL 的基础上，
        对高度冗余图像施加软惩罚，避免重复视觉证据被过度累积。

    输入：
        image_features: [N, D]

    输出：
        pooled:     [D]
        image_mask: [1]
        attn:       [N]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        redundancy_lambda: float = 1.0,
        redundancy_threshold: float = 0.75,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.redundancy_lambda = redundancy_lambda
        self.redundancy_threshold = redundancy_threshold
        self.eps = eps

        # Gated Attention MIL 打分分支
        self.norm = nn.LayerNorm(feature_dim)

        self.V = nn.Linear(feature_dim, hidden_dim)
        self.U = nn.Linear(feature_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def _compute_redundancy(
        self,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每张图像的冗余程度。

        image_features:
            [N, D]

        返回：
            redundancy: [N]

        计算逻辑：
            1. 对图像特征做 L2 normalize；
            2. 计算两两 cosine similarity；
            3. 去掉对角线，也就是去掉图像自己和自己的相似度；
            4. 只保留超过 redundancy_threshold 的高相似部分；
            5. 对每张图像求平均冗余分数。
        """
        n = image_features.shape[0]
        device = image_features.device

        if n <= 1:
            return torch.zeros(
                n,
                dtype=torch.float32,
                device=device,
            )

        # [N, D]
        norm_features = F.normalize(
            image_features,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        # [N, N]
        sim_matrix = torch.matmul(
            norm_features,
            norm_features.transpose(0, 1),
        )

        # 去掉对角线：避免自己和自己相似度为 1
        eye_mask = torch.eye(
            n,
            dtype=torch.bool,
            device=device,
        )

        sim_matrix = sim_matrix.masked_fill(
            eye_mask,
            0.0,
        )

        # 只惩罚超过阈值的相似部分。
        # 例如 threshold=0.75，则 sim <= 0.75 的图像对不认为是明显冗余。
        redundancy_matrix = F.relu(
            sim_matrix - self.redundancy_threshold
        )

        # [N]
        redundancy = redundancy_matrix.sum(dim=1) / max(n - 1, 1)

        return redundancy

    def forward(
        self,
        image_features: Optional[torch.Tensor],
        query_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        image_features:
            [N, D]

        query_feat:
            预留接口。当前模块不使用文本或其他 query。
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

        # =========================
        # 1. Gated Attention MIL 原始打分
        # =========================
        x = self.norm(image_features)  # [N, D]

        h_v = torch.tanh(self.V(x))       # [N, H]
        h_u = torch.sigmoid(self.U(x))    # [N, H]
        h = h_v * h_u                     # [N, H]

        raw_scores = self.w(h).squeeze(-1)  # [N]

        # =========================
        # 2. 相似性感知冗余分数
        # =========================
        redundancy = self._compute_redundancy(
            image_features=image_features,
        )  # [N]

        # =========================
        # 3. 冗余抑制后的最终打分
        # =========================
        final_scores = raw_scores - self.redundancy_lambda * redundancy

        # =========================
        # 4. Softmax 得到注意力权重
        # =========================
        attn = F.softmax(
            final_scores,
            dim=0,
        )  # [N]

        # =========================
        # 5. 加权求和得到检查级图像特征
        # =========================
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
        """
        query_feat 是预留接口。
        当前普通 Attention Pooling 不使用它。
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


class MeanImagePooling(nn.Module):
    """
    平均池化 baseline。

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

        image_features = image_features.float().to(device)  # [N, D]

        n = image_features.shape[0]

        pooled = image_features.mean(dim=0)  # [D]

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
        sra_gated_mil
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

    if pooling_type in [
        "sra_gated_mil",
        "redundancy_gated_mil",
        "similarity_redundancy_gated_mil",
        "srs_gated_mil",
    ]:
        return SimilarityAwareRedundancySuppressedGatedAttentionPooling(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            redundancy_lambda=1.0,
            redundancy_threshold=0.75,
        )

    raise ValueError(f"Unknown image pooling type: {pooling_type}")