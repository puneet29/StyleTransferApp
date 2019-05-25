# Importing the resources
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from transform import TransformNet

# Command line arguments
argparser = argparse.ArgumentParser(description="An Image Style Transfer tool")
argparser.add_argument('-s', '--style', type=str, default='images/style.jpg',
                       help='Path to style image')
argparser.add_argument('-c', '--content', type=str, default='images/content.jpg',
                       help='Path to content image')

args = argparser.parse_args()


# SETTINGS
SAVED_MODEL = 'model/'
SAVE_MODEL_EVERY = 400
LR = 0.003
STYLE_WEIGHT = 1e11
CONTENT_WEIGHT = 500
EPOCHS = 1
IMAGE_SIZE_MAX = 256
BATCH_SIZE = 4
STYLE_IMAGE_PATH = args.style
CONTENT_IMAGE_PATH = args.content


def load_image(path):
    return(cv2.imread(path))


def tensor2image(img, max_size=None):
    image = img.convert('RGB')
    if(max_size == None):
        tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    else:
        if(max(image.size) > max_size):
            size = max_size
        else:
            size = max(image.size)
        tensor = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    tensor = tensor(img).unsqueeze(dim=0)
    return(tensor)


def gram_matrix(tensor):
    # needs testing
    _, d, h, w = tensor.shape
    tensor = tensor.view(d, h*w)

    gram = torch.mm(tensor, tensor.t())
    return(gram)


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
    TransformerNet = TransformNet().to(device)
    vgg = models.vgg19(pretrained=True)

    # Get Style Features
    style_image = load_image(STYLE_IMAGE_PATH)
    style_tensor = tensor2image(style_image).to(device)
    B, C, H, W = style_tensor.shape
    style_features = vgg(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = gram_matrix(value)

    # Optimizer
    optimizer = optim.Adam(TransformerNet.parameters(), lr=LR)
