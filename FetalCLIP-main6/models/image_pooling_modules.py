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



class RepresentativeRedundancySuppressedGatedAttentionPooling(nn.Module):
    """
    Representative Redundancy Suppression Gated Attention Pooling.

    中文名：
        代表性选择式冗余抑制门控注意力池化模块。

    设计目标：
        当单次检查中存在多张高度相似图像时，
        不把相似图像全部压低，
        而是在相似图像组内保留 raw_score 更高的代表图，
        主要压低 raw_score 较低的重复图。

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
        score_temperature: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.redundancy_lambda = redundancy_lambda
        self.redundancy_threshold = redundancy_threshold
        self.score_temperature = score_temperature
        self.eps = eps

        # Gated Attention MIL scoring branch
        self.norm = nn.LayerNorm(feature_dim)

        self.V = nn.Linear(feature_dim, hidden_dim)
        self.U = nn.Linear(feature_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def _compute_representative_redundancy(
        self,
        image_features: torch.Tensor,
        raw_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算代表性选择式冗余分数。

        image_features:
            [N, D]

        raw_scores:
            [N]

        返回：
            redundancy: [N]

        核心逻辑：
            对图像 i，如果存在与它高度相似的图像 j，
            且 raw_score_j > raw_score_i，
            那么 i 更可能是重复图，应被惩罚。

            如果 i 是相似组内 raw_score 最高的代表图，
            它不会因为低分重复图而被明显惩罚。
        """
        n = image_features.shape[0]
        device = image_features.device

        if n <= 1:
            return torch.zeros(
                n,
                dtype=torch.float32,
                device=device,
            )

        # =========================
        # 1. 计算两两余弦相似度
        # =========================
        norm_features = F.normalize(
            image_features,
            p=2,
            dim=-1,
            eps=self.eps,
        )  # [N, D]

        sim_matrix = torch.matmul(
            norm_features,
            norm_features.transpose(0, 1),
        )  # [N, N]

        eye_mask = torch.eye(
            n,
            dtype=torch.bool,
            device=device,
        )

        # 去掉自己和自己的相似度
        sim_matrix = sim_matrix.masked_fill(
            eye_mask,
            0.0,
        )

        # 只考虑超过阈值的相似图像对
        sim_excess = F.relu(
            sim_matrix - self.redundancy_threshold
        )  # [N, N]

        # =========================
        # 2. 判断 j 是否比 i 更适合作为代表图
        # =========================
        # row i, col j:
        # score_diff[i, j] = raw_score_j - raw_score_i
        score_i = raw_scores.unsqueeze(1)  # [N, 1]
        score_j = raw_scores.unsqueeze(0)  # [1, N]

        score_diff = score_j - score_i  # [N, N]

        # soft version:
        # 如果 raw_score_j 明显高于 raw_score_i，则 higher_weight 接近 1；
        # 如果 raw_score_j 低于 raw_score_i，则 higher_weight 接近 0。
        higher_weight = torch.sigmoid(
            score_diff / self.score_temperature
        )  # [N, N]

        higher_weight = higher_weight.masked_fill(
            eye_mask,
            0.0,
        )

        # =========================
        # 3. 只惩罚“和更高分代表图相似”的图像
        # =========================
        redundancy_matrix = sim_excess * higher_weight  # [N, N]

        redundancy = redundancy_matrix.sum(dim=1) / max(n - 1, 1)  # [N]

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
            预留接口。当前模块不使用文本特征。
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
        # 2. 代表性选择式冗余分数
        # =========================
        redundancy = self._compute_representative_redundancy(
            image_features=image_features,
            raw_scores=raw_scores,
        )  # [N]

        # =========================
        # 3. 只压低低分重复图，保留高分代表图
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
        # 5. 加权求和
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


class ReliabilityCalibratedConsensusPooling(nn.Module):
    """
    Reliability-Calibrated Consensus Pooling.

    The module keeps mean pooling as a stable consensus baseline, then learns a
    small residual correction from per-image reliability, consensus consistency,
    and complementary residual evidence. It does not use text features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        initial_residual_ratio: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.eps = eps

        self.norm = nn.LayerNorm(feature_dim)
        self.score_net = nn.Sequential(
            nn.LayerNorm(feature_dim * 4 + 3),
            nn.Linear(feature_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        ratio = min(max(initial_residual_ratio, 1e-4), 1.0 - 1e-4)
        logit = torch.logit(torch.tensor(ratio, dtype=torch.float32))
        self.residual_logit = nn.Parameter(logit)

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

        image_features = image_features.float().to(device)  # [N, D]

        if image_features.shape[0] == 1:
            pooled = image_features.squeeze(0)
            image_mask = torch.tensor(
                [1.0],
                dtype=torch.float32,
                device=device,
            )
            attn = torch.ones(
                1,
                dtype=torch.float32,
                device=device,
            )
            return pooled, image_mask, attn

        consensus = image_features.mean(dim=0)  # [D]
        x_normed = self.norm(image_features)
        c_normed = self.norm(consensus).unsqueeze(0).expand_as(x_normed)

        residual = x_normed - c_normed
        interaction = x_normed * c_normed

        consistency = F.cosine_similarity(
            image_features,
            consensus.unsqueeze(0),
            dim=-1,
            eps=self.eps,
        ).unsqueeze(-1)

        residual_norm = torch.linalg.vector_norm(
            residual,
            ord=2,
            dim=-1,
            keepdim=True,
        ) / (self.feature_dim ** 0.5)

        pairwise_sim = torch.matmul(
            F.normalize(image_features, p=2, dim=-1, eps=self.eps),
            F.normalize(image_features, p=2, dim=-1, eps=self.eps).transpose(0, 1),
        )
        n = image_features.shape[0]
        pairwise_sim = pairwise_sim.masked_fill(
            torch.eye(n, dtype=torch.bool, device=device),
            0.0,
        )
        redundancy = pairwise_sim.clamp(min=0.0).sum(dim=1, keepdim=True) / max(n - 1, 1)
        uniqueness = 1.0 - redundancy

        score_input = torch.cat(
            [
                x_normed,
                c_normed,
                residual,
                interaction,
                consistency,
                residual_norm,
                uniqueness,
            ],
            dim=-1,
        )

        scores = self.score_net(score_input).squeeze(-1)
        scores = scores + consistency.squeeze(-1) + 0.25 * uniqueness.squeeze(-1)
        attn = F.softmax(scores, dim=0)

        calibrated = torch.sum(
            image_features * attn.unsqueeze(-1),
            dim=0,
        )
        residual_ratio = torch.sigmoid(self.residual_logit)
        pooled = (1.0 - residual_ratio) * consensus + residual_ratio * calibrated

        image_mask = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )

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
        rcc_pooling
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
    
    if pooling_type in [
        "rrs_gated_mil",
        "representative_redundancy_gated_mil",
        "representative_suppression_gated_mil",
    ]:
        return RepresentativeRedundancySuppressedGatedAttentionPooling(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            redundancy_lambda=1.0,
            redundancy_threshold=0.75,
            score_temperature=0.1,
        )

    if pooling_type in [
        "rcc_pooling",
        "rcc",
        "reliability_calibrated_consensus",
        "reliability_consensus",
    ]:
        return ReliabilityCalibratedConsensusPooling(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    raise ValueError(f"Unknown image pooling type: {pooling_type}")
