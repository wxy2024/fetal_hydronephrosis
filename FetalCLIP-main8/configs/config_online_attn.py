import os
from pathlib import Path


# =========================
# 模块组合配置
# =========================

# 图像聚合方式：
# mean / attention / gated_mil / text_cross_attention / sra_gated_mil / rrs_gated_mil / rcc_pooling / vcm_pooling / pgs_pooling
IMAGE_POOLING_TYPE = "rcc_pooling"

# 检查级融合方式：
# direct_concat / gated_multimodal / film_gated_multimodal / transformer_token
VISIT_FUSION_TYPE = "direct_concat"

FUSION_HIDDEN_DIM = None

TEMPORAL_BACKBONE = "gru"
# mean_pooling / simple_fusion / gru / gru_d / risk_gru / td_conv_gru / td_conv_gru_no_conv / bit_gru / tripg_gru / ca_grud / psalm_grud / contiformer

# Input modality ablation:
#   multimodal: image + text + structural/time features
#   image_only: image + structural/time features (text is not encoded)
#   text_only:  text + structural/time features (images are not encoded and
#               image availability is not exposed to the classifier)
INPUT_MODALITY = os.environ.get("INPUT_MODALITY", "multimodal").strip().lower()
VALID_INPUT_MODALITIES = {"multimodal", "image_only", "text_only"}
if INPUT_MODALITY not in VALID_INPUT_MODALITIES:
    raise ValueError(
        f"Unknown INPUT_MODALITY={INPUT_MODALITY!r}. "
        f"Expected one of {sorted(VALID_INPUT_MODALITIES)}."
    )

# Temporal-information ablation. All modes use the same original cohort, which
# is first restricted to cases with at least TEMPORAL_MIN_VISITS examinations.
# ``shuffled`` is intentionally defined as ``no_explicit_time + shuffled
# visits`` so that negative intervals or stage/age clues cannot confound the
# effect of destroying chronological order.
TEMPORAL_EXPERIMENT = os.environ.get(
    "TEMPORAL_EXPERIMENT", "baseline"
).strip().lower()
VALID_TEMPORAL_EXPERIMENTS = {
    "baseline",
    "first_only",
    "last_only",
    "shuffled",
    "no_explicit_time",
}
if TEMPORAL_EXPERIMENT not in VALID_TEMPORAL_EXPERIMENTS:
    raise ValueError(
        f"Unknown TEMPORAL_EXPERIMENT={TEMPORAL_EXPERIMENT!r}. "
        f"Expected one of {sorted(VALID_TEMPORAL_EXPERIMENTS)}."
    )

TEMPORAL_MIN_VISITS = int(os.environ.get("TEMPORAL_MIN_VISITS", "2"))
if TEMPORAL_MIN_VISITS < 2:
    raise ValueError("TEMPORAL_MIN_VISITS must be at least 2 for this experiment.")

# This seed changes only the visit permutation. It does not change model
# initialization, folds, sampling, or the training seed.
TEMPORAL_SHUFFLE_SEED = int(os.environ.get("TEMPORAL_SHUFFLE_SEED", "42"))
NO_EXPLICIT_TIME_EXPERIMENTS = {"no_explicit_time", "shuffled"}
USE_EXPLICIT_TIME = TEMPORAL_EXPERIMENT not in NO_EXPLICIT_TIME_EXPERIMENTS


# =========================
# 1. 数据路径配置
# =========================
RAW_ROOT = Path(r"E:\wxy_codes\dataset_hydro_1234\raw")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output6_rebuttal"
#RESULT_DIR = OUTPUT_ROOT / "results_10_new1and2_fig_rrs_gated_mil_visit_film_gated_multimodal_contiformer"
#结果文件命名改这两个
EXP_ID = 1012
#EXP_ID = int(os.environ.get("EXP_ID", 34))


RESULT_DIR = OUTPUT_ROOT / (
    f"rebuttal_{EXP_ID}_dataset_1234"
    f"_fig_{IMAGE_POOLING_TYPE}"
    f"_visit_{VISIT_FUSION_TYPE}"
    f"_{TEMPORAL_BACKBONE}"
    f"_modality_{INPUT_MODALITY}"
    f"_temporal_{TEMPORAL_EXPERIMENT}"
    + (
        f"_shuffle_seed_{TEMPORAL_SHUFFLE_SEED}"
        if TEMPORAL_EXPERIMENT == "shuffled"
        else ""
    )
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
SEED = 60
# epoch 没标注就默认是50
# Nested CV treats this as the maximum number of epochs considered by each
# inner fold.  The selected epoch is then used to retrain on the complete
# outer-training fold before the outer test fold is evaluated once.
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "50"))
INNER_N_SPLITS = int(os.environ.get("INNER_N_SPLITS", "3"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "10"))
EARLY_STOPPING_MIN_DELTA = float(
    os.environ.get("EARLY_STOPPING_MIN_DELTA", "1e-4")
)

if NUM_EPOCHS < 1:
    raise ValueError("NUM_EPOCHS must be at least 1.")
if INNER_N_SPLITS < 2:
    raise ValueError("INNER_N_SPLITS must be at least 2.")
if EARLY_STOPPING_PATIENCE < 1:
    raise ValueError("EARLY_STOPPING_PATIENCE must be at least 1.")
if EARLY_STOPPING_MIN_DELTA < 0:
    raise ValueError("EARLY_STOPPING_MIN_DELTA must be non-negative.")
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
BATCH_SIZE = 32

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
