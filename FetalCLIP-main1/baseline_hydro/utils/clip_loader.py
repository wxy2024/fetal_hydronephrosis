import json
import open_clip
from config import PATH_FETALCLIP_CONFIG, PATH_FETALCLIP_WEIGHT, ARCH_NAME

def load_fetalclip(device="cuda"):
    with open(PATH_FETALCLIP_CONFIG, "r", encoding="utf-8") as f:
        config_fetalclip = json.load(f)

    open_clip.factory._MODEL_CONFIGS[ARCH_NAME] = config_fetalclip

    model, _, preprocess = open_clip.create_model_and_transforms(
        ARCH_NAME,
        pretrained=PATH_FETALCLIP_WEIGHT
    )
    tokenizer = open_clip.get_tokenizer(ARCH_NAME)

    model.eval()
    model = model.to(device)

    return model, preprocess, tokenizer