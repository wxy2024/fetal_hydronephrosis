# models/baseline_model.py
import torch
import torch.nn as nn


class SimpleSideClassifier(nn.Module):
    """
    单侧肾手术预测分类器。

    输入：
        side_feature: [B, input_dim]

    输出：
        logits: [B]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        logits = self.classifier(x).squeeze(-1)
        return logits