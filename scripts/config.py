import torch

class Config:
    SEED = 42

    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "resnet50.a1_in1k" # "efficientnet_b3"
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "layer4" # "blocks.5|conv_head|bn2"
    BATCH_SIZE = 32
    TEXT_LR = 1e-4
    IMAGE_LR = 1e-4
    REGRES_LR = 1e-3
    EPOCHS = 12
    HIDDEN_DIM = 256
    IMAGE_PATH = "data/images/"
    DF_GENERAL_PATH = "data/dish.csv"
    DF_PRODUCT_PATH = "data/ingredients.csv"
    SAVE_PATH = "best_model.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_config() -> Config:
    """
    Returns the default configuration. Modify fields in the notebook before calling train(cfg).
    """
    return Config()