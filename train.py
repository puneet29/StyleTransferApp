# Importing the resources
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from transform import TransformNet
from utils import gram_matrix, image2tensor, load_image, plot_loss_hist, saveimg, tensor2image
from vgg import VGG16


def check_paths(args):
    try:
        if(not os.path.exists(args.save_model_dir)):
            os.makedirs(args.save_model_dir)
        if(args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir))):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):

    # Select GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setting seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Datasets and Dataloaders
    train_dataset = datasets.ImageFolder(args.dataset, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load Network
    TransformerNet = TransformNet().to(device)
    vgg = VGG16().to(device)

    # Optimizer
    optimizer = optim.Adam(TransformerNet.parameters(), lr=args.lr)


# Command line arguments
arg_parser = argparse.ArgumentParser(
    description="parser for fast-neural-style")

arg_parser.add_argument("--epochs", type=int, default=2,
                        help="number of training epochs, default is 2")
arg_parser.add_argument("--batch-size", type=int, default=4,
                        help="batch size for training, default is 4")
arg_parser.add_argument("--dataset", type=str, required=True,
                        help="path to training dataset, the path should point to a folder "
                        "containing another folder with all the training images")
arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                        help="path to style-image")
arg_parser.add_argument("--save-model-dir", type=str, required=True,
                        help="path to folder where trained model will be saved.")
arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                        help="path to folder where checkpoints of trained models will be saved")
arg_parser.add_argument("--image-size", type=int, default=256,
                        help="size of training images, default is 256 X 256")
arg_parser.add_argument("--style-size", type=int, default=None,
                        help="size of style-image, default is the original size of style image")
arg_parser.add_argument("--cuda", type=int, required=True,
                        help="set it to 1 for running on GPU, 0 for CPU")
arg_parser.add_argument("--seed", type=int, default=42,
                        help="random seed for training")
arg_parser.add_argument("--content-weight", type=float, default=1e5,
                        help="weight for content-loss, default is 1e5")
arg_parser.add_argument("--style-weight", type=float, default=1e10,
                        help="weight for style-loss, default is 1e10")
arg_parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate, default is 1e-3")
arg_parser.add_argument("--log-interval", type=int, default=500,
                        help="number of images after which the training loss is logged, default is 500")
arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                        help="number of batches after which a checkpoint of the trained model will be created")

args = arg_parser.parse_args()

if args.cuda and not torch.cuda.is_available():
    print("ERROR: cuda is not available, try running on CPU")
    sys.exit(1)

check_paths(args)
train(args)
