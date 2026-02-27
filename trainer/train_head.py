import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.metrics import accuracy
from configs.config import *


def train_head(model, train_loader, val_loader):
	for param in model.parameters():
		param.requires_grad = False
	for param in model.head.parameters():
		param.reuires_grad = True
		
	
	optimizer = Adam(model.head.parameters(), lr = LR_HEAD)
	criterion = CrossEntropyLoss()
	model.to(DEVICE)
	
	for epochs in range(NUM_EPOCHS_HEAD):
		model.train()
		for x,y in train_loader:
			x,y = x.to(DEVICE), y.to(DEVICE)
			optimizer.zero_grad()
			preds = model(x)
			loss = criterion(preds, y)
			loss.backward()
			optimizer.step()
			
		
	print(f"[Head] Epoch {epochs + 1}/{NUM_EPOCHS_HEAD} completed")
	
	
	
