import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FETALCLIP_ROOT = os.path.dirname(PROJECT_ROOT)

PATH_FETALCLIP_WEIGHT = os.path.join(FETALCLIP_ROOT, "FetalCLIP_weights.pt")
PATH_FETALCLIP_CONFIG = os.path.join(FETALCLIP_ROOT, "FetalCLIP_config.json")
ARCH_NAME = "FetalCLIP"

RAW_ROOT = r"G:\wxy_codes\Fetal_hydronephrosis\fetal_hydro_dataset\raw"

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
FEATURE_DIR = os.path.join(OUTPUT_ROOT, "features")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")

INDEX_CSV_PATH = os.path.join(FEATURE_DIR, "visit_feature_index.csv")
SPLIT_JSON_PATH = os.path.join(LOG_DIR, "data_split.json")
BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_mlp.pt")
TEST_METRICS_JSON_PATH = os.path.join(LOG_DIR, "test_metrics.json")

RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2

DEVICE = "cuda"