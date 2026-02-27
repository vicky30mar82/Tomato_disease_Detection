import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from configs.config import *

def get_dataloaders():
	train_tfms = transforms.Compose([
		transforms.RandomResizedCrop(IMG_SIZE),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(15),
		transforms.ColorJitter(0.3,0.3,0.3),
		transforms.ToTensor()
	])
	
	test_tfms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(IMG_SIZE),
		transforms.ToTensor()
	])
	
	train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tfms)
	val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), train_tfms)
	
	train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
	val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = False)
	
	return train_loader, val_loader, len(train_ds.classes)
