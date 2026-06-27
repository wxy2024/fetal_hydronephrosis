from pathlib import Path


# =========================
# 模块组合配置
# =========================

# 图像聚合方式：
# mean / attention / gated_mil / text_cross_attention / sra_gated_mil / rrs_gated_mil / rcc_pooling
IMAGE_POOLING_TYPE = "rrs_gated_mil"

# 检查级融合方式：
# direct_concat / gated_multimodal / film_gated_multimodal / transformer_token
VISIT_FUSION_TYPE = "film_gated_multimodal"

FUSION_HIDDEN_DIM = None

TEMPORAL_BACKBONE = "gru"
# simple_fusion / gru / ca_grud / psalm_grud / contiformer


# =========================
# 1. 数据路径配置
# =========================
RAW_ROOT = Path(r"G:\wxy_codes\dataset_hydro_123\raw")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
#RESULT_DIR = OUTPUT_ROOT / "results_10_new1and2_fig_rrs_gated_mil_visit_film_gated_multimodal_contiformer"
#结果文件命名改这两个
EXP_ID = 21


RESULT_DIR = OUTPUT_ROOT / (
    f"results_{EXP_ID}_dataset_123"
    f"_fig_{IMAGE_POOLING_TYPE}"
    f"_visit_{VISIT_FUSION_TYPE}"
    f"_{TEMPORAL_BACKBONE}"
)

LOG_DIR = OUTPUT_ROOT / "logs"

# =========================
# 2. FetalCLIP 配置
# =========================
FETALCLIP_CONFIG_PATH = PROJECT_ROOT / "FetalCLIP_config.json"
FETALCLIP_WEIGHT_PATH = PROJECT_ROOT / "FetalCLIP_weights.pt"
FETALCLIP_MODEL_NAME = "FetalCLIP"

# =========================
# 3. 数据读取配置
# =========================
DEVICE = "cuda"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

USE_PRENATAL = True
USE_POSTNATAL = True

MAX_REPORT_CHARS = 512

# =========================
# 4. 训练配置
# =========================
SEED = 42
# epoch 没标注就默认是50
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
BATCH_SIZE = 4

# 是否使用 pos_weight 处理类别不平衡。
# 你之前发现 USE_POS_WEIGHT=False 时 GRU 结果更稳，所以这里默认 False。
USE_POS_WEIGHT = False

# =========================
# 5. 模型配置
# =========================
ATTN_HIDDEN_DIM = 256
GRU_HIDDEN_DIM = 128
GRU_NUM_LAYERS = 1
GRU_BIDIRECTIONAL = True

# FetalCLIP 冻结时，在训练前把图像/文本特征提取到内存，不保存为 .pt 文件。
# 这样比每个 epoch 重新编码更快，且不改变训练结果。
USE_MEMORY_CACHE = True

# 是否在预测结果 csv 中保存每次检查内各图像的 attention 权重。
SAVE_ATTENTION = True



# =========================
# 6. ContiFormer 配置
# =========================
# TEMPORAL_BACKBONE = "contiformer"

# CONTIFORMER_D_MODEL = 32
# CONTIFORMER_D_INNER = 64
# CONTIFORMER_N_LAYERS = 1
# CONTIFORMER_N_HEAD = 1
# CONTIFORMER_D_K = 32
# CONTIFORMER_D_V = 32

# CONTIFORMER_DROPOUT = DROPOUT

# CONTIFORMER_METHOD_ODE = "rk4"
# CONTIFORMER_ATOL_ODE = 1e-3
# CONTIFORMER_RTOL_ODE = 1e-3
