import torch

DATA_DIR = "data"
BATCH_SIZE = 16
NUM_EPOCHS_HEAD = 5
NUM_EPOCHS_FINE = 20
LR_HEAD = 1e-3
LR_FINE = 3e-5
IMG_SIZE = 224
DEVICE = "cuda" "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vit_base_patch16_224"
