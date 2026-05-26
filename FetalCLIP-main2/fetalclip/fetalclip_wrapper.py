# fetalclip/fetalclip_wrapper.py
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image

import open_clip


class FetalCLIPFeatureExtractor:
    """
    FetalCLIP 特征提取器。

    功能：
    1. 加载 FetalCLIP 图像-文本编码器；
    2. 冻结模型；
    3. 提取图像特征；
    4. 提取文本特征。

    注意：
    如果你的 FetalCLIP 官方代码加载方式不同，
    主要修改 _load_model() 这一部分即可。
    """

    def __init__(
        self,
        model_name: str,
        config_path: Path,
        weight_path: Path,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.config_path = Path(config_path)
        self.weight_path = Path(weight_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model, self.preprocess, self.tokenizer = self._load_model()

        self.model.eval()
        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

    def _load_model(self):
        """
        加载 FetalCLIP。

        常见情况：
        - FetalCLIP_config.json 中保存 open_clip 模型结构；
        - FetalCLIP_weights.pt 中保存权重；
        - 需要先注册到 open_clip.factory._MODEL_CONFIGS。
        """

        if not self.config_path.exists():
            raise FileNotFoundError(f"找不到 FetalCLIP config: {self.config_path}")

        if not self.weight_path.exists():
            raise FileNotFoundError(f"找不到 FetalCLIP weight: {self.weight_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)

        # 注册自定义模型配置
        open_clip.factory._MODEL_CONFIGS[self.model_name] = model_cfg

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=str(self.weight_path)
        )

        tokenizer = open_clip.get_tokenizer(self.model_name)

        return model, preprocess, tokenizer

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        编码多张图像。

        返回：
            shape = [N, D]

        如果 image_paths 为空，返回 None。
        """

        if len(image_paths) == 0:
            return None

        images = []

        for p in image_paths:
            p = Path(p)
            try:
                img = Image.open(p).convert("RGB")
                img_tensor = self.preprocess(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"[警告] 图像读取失败，已跳过: {p}, error={e}")

        if len(images) == 0:
            return None

        images = torch.stack(images, dim=0).to(self.device)

        image_features = self.model.encode_image(images)
        image_features = image_features.float()

        # CLIP 常用做法：L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return image_features.cpu()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码单段文本。

        返回：
            shape = [D]
        """

        if text is None or len(text.strip()) == 0:
            text = "No report."

        tokens = self.tokenizer([text]).to(self.device)

        text_features = self.model.encode_text(tokens)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return text_features.squeeze(0).cpu()