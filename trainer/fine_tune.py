import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.metrics import accuracy
from configs.config import *


def fine_tune(model, train_loader, val_loader):
	for param in model.parameters():
		param.requires_grad = True
		
	optimizer = Adam(model.parameters(), lr = LR_RATE)
	criterion = CrossEntropyLoss()
	model.to(DEVICE)
	
	for epoch in range(NUM_EPOCHS_FINE):
		model.train()
		for x,y in train_loader:
			x,y = x.to(DEVICE), y.to(DEVICE)
			optimizer.zero_grad()
			preds = model(x)
			loss = criterion(preds,y)
			loss.backward()
			optimizer.step()
			
		
	print(f"[Fine Tune] Epochs {epochs + 1}/{NUM_EPOCHS_FINE} completed")
