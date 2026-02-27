import timm
import torch.nn as nn
from configs.config import *

def create_vit(num_classes):
	model = timm.create_model(MODEL_NAME, pretrained = True)
	model.head = nn.Linear(model.head.in_features, num_classes)
	return model
