# configs/config.py
from pathlib import Path


# =========================
# 1. 数据路径配置
# =========================

# 这里改成你的 raw 文件夹路径
RAW_ROOT = Path(r"G:\wxy_codes\Fetal_hydronephrosis\dataset\3合并12批数据\raw")

# 输出目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_ROOT / "features_4"
RESULT_DIR = OUTPUT_ROOT / "results_12_features4+03_gru_nobidirec_epo50_DR"
LOG_DIR = OUTPUT_ROOT / "logs"


# =========================
# 2. FetalCLIP 配置
# =========================

# 这里改成你本地下载的 FetalCLIP 配置文件和权重文件路径
# 示例：
# FETALCLIP_CONFIG_PATH = Path(r"G:\wxy_codes\FetalCLIP\FetalCLIP_config.json")
# FETALCLIP_WEIGHT_PATH = Path(r"G:\wxy_codes\FetalCLIP\FetalCLIP_weights.pt")

FETALCLIP_CONFIG_PATH = Path(r"G:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main2\FetalCLIP_config.json")
FETALCLIP_WEIGHT_PATH = Path(r"G:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main2\FetalCLIP_weights.pt")

# FetalCLIP 模型名称，通常需要和 config 中注册的名称保持一致
FETALCLIP_MODEL_NAME = "FetalCLIP"


# =========================
# 3. 运行配置
# =========================

DEVICE = "cuda"  # 没有 GPU 就改成 "cpu"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# 是否使用出生前 + 出生后全部检查
USE_PRENATAL = True
USE_POSTNATAL = True

# 单个 report.txt 最大文本长度。
# FetalCLIP 文本编码器本身可能有 token 长度限制，所以这里先做字符级截断，避免极长报告报错。
MAX_REPORT_CHARS = 512


# =========================
# 4. 训练配置
# =========================

SEED = 42

NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# 对于 12 个病例的小样本，batch_size 可以直接设为训练集中全部样本数；
# 这里先写成 8，后面 DataLoader 会自动处理。
BATCH_SIZE = 8

# 是否使用 pos_weight 处理类别不平衡
USE_POS_WEIGHT = False