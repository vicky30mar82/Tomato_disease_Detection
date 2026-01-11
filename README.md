# Tomato_disease_Detection
Vision Transformer–based tomato leaf disease detection using transfer learning on small datasets with a clean, modular PyTorch pipeline.

This repository implements an end-to-end tomato plant disease classification system using a pretrained Vision Transformer (ViT) in PyTorch, optimized for small datasets through transfer learning, strong data augmentation, and staged fine-tuning.

Overview

The project focuses on applying Vision Transformers to agricultural image classification where data availability is limited. It follows a production-style, modular code structure and supports reproducible training and evaluation.

Project Structure
tomato_vit/
│
├── data/                     Dataset (train/val/test)
├── configs/                  Training configuration
├── dataset/                  Dataloader and augmentations
├── models/                   ViT model definition
├── trainer/                  Training and fine-tuning logic
├── utils/                    Metrics and helper utilities
├── main.py                   Training entry point
└── requirements.txt

Dataset

Recommended datasets:

PlantVillage Tomato Leaf Diseases
https://www.kaggle.com/datasets/emmarex/plantdisease

Roboflow Tomato Disease Datasets
https://universe.roboflow.com

Expected directory format:

data/
 ├── train/
 ├── val/
 └── test/
     └── class_name/

Installation
pip install -r requirements.txt


Requirements:

Python 3.8 or higher

PyTorch

torchvision

timm

Training

Run the complete training pipeline including head training and full fine-tuning:

python main.py


The trained model will be saved as:

vit_tomato_disease.pth

Training Strategy

The training process follows these steps:

Initialize a pretrained Vision Transformer

Freeze the backbone and train the classification head

Unfreeze the full network for fine-tuning

Apply strong data augmentation to reduce overfitting

Evaluation

The current implementation tracks training progress and accuracy. It is designed to be extended with additional evaluation metrics such as precision, recall, confusion matrix, and false positive analysis.

Extensions

The project can be easily extended to include:

False positive reduction using threshold tuning and calibration

Test-time augmentation

Grad-CAM visualizations

Inference API using FastAPI

Integration with HuggingFace ViT

Edge deployment optimizations

Use Cases

Smart agriculture

Early crop disease detection

Vision Transformer experimentation on small datasets

Interview-ready deep learning project
