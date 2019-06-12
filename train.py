# Importing the resources
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformer import TransformNet
from utils import gram_matrix, load_image, normalize_batch
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
    device = torch.device('cuda' if args.cuda else 'cpu')

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
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load Network
    transformer = TransformNet().to(device)
    vgg = VGG16(False).to(device)

    # Optimizer
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)

    # Loss function
    mse_loss = nn.MSELoss()

    # Style features
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    # Repeat tensor along the specified dimensions
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(x) for x in features_style]

    # Training loop
    for epoch in range(args.epochs):
        transformer.train()
        total_content_loss = 0.0
        total_style_loss = 0.0
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):

            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            # Output from transformer network -> y
            x = x.to(device)
            y = transformer(x)

            # Normalize batches (y-> output from transformer, x-> raw input)
            y = normalize_batch(y)
            x = normalize_batch(x)

            # Output from vgg model
            features_y = vgg(y)
            features_x = vgg(x)

            # Calculate content loss
            content_loss = args.content_weight * \
                mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Calculate style loss
            style_loss = 0.0
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            # Calculate batch loss
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # Calculate total loss
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()

            if((batch_id + 1) % args.log_interval == 0):
                print(
                    f'{time.ctime()}\tEpoch {epoch+1}:\t[{count}/{len(train_dataset)}]\tcontent: {total_content_loss / batch_id + 1}\tstyle: {total_style_loss / batch_id + 1}\ttotal: {(total_content_loss + total_style_loss) / (batch_id + 1)}')

            if(args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0):
                transformer.eval().cpu()
                ckpt_model_filename = 'ckpt_epoch_' + \
                    str(epoch) + "_batch_id_" + str(batch_id + 1) + '.pth'
                ckpt_model_path = os.path.join(
                    args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device)
    # Save model
    transformer.eval().cpu()
    save_model_filename = 'epoch_' + str(args.epochs) + '_' + str(time.ctime()).replace(
        ' ', '_') + '_' + str(args.content_weight) + '_' + str(args.style_weight) + '.model'
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print('\nModel trained! It is saved at:', save_model_path)


# Command line arguments
arg_parser = argparse.ArgumentParser(
    description="parser for fast-neural-style-training")

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
