# Importing the resources
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets

# SETTINGS
SAVED_MODEL = 'model/'
SAVE_MODEL_EVERY = 400
LR = 0.003
STYLE_WEIGHT = 1e11
CONTENT_WEIGHT = 500
EPOCHS = 1
IMAGE_SIZE_MAX = 256
BATCH_SIZE = 4


def train():

    # Select GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_MAX),
        transforms.CenterCrop(IMAGE_SIZE_MAX),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Datasets and Dataloaders
    train_dataset = datasets.ImageFolder('dataset', transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Network
    TransformNet = 
