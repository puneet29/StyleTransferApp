# Importing the resources
import argparse
import sys

import torch
from torchvision import transforms
from transformer import TransformNet
from utils import load_image, match_size, save_image


def stylize(args):

    # Select GPU if available
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Load content image
    content_image = load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    # Set requires_grad to False
    with torch.no_grad():
        style_model = TransformNet()
        state_dict = torch.load(args.model)

        # Load the model's learnt params
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        # Output image
        output = style_model(content_image).cpu()

    content_image = match_size(content_image, output)
    weighted_output = output * args.style_strength + \
        (content_image * (1 - args.style_strength))
    save_image(args.output_image, weighted_output[0])


# Command line arguments
eval_arg_parser = argparse.ArgumentParser(
    description="parser for fast-neural-style-evaluation")
eval_arg_parser.add_argument("--content-image", type=str, required=True,
                             help="path to content image you want to stylize")
eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                             help="factor for scaling down the content image")
eval_arg_parser.add_argument("--output-image", type=str, required=True,
                             help="path for saving the output image")
eval_arg_parser.add_argument("--model", type=str, required=True,
                             help="saved model to be used for stylizing the image.")
eval_arg_parser.add_argument("--cuda", type=int, required=True,
                             help="set it to 1 for running on GPU, 0 for CPU")
eval_arg_parser.add_argument("--style-strength", type=float, default=1.0,
                             help="set between 0 and 1, the strength of style, default 1.0")

args = eval_arg_parser.parse_args()

if(args.cuda and not torch.cuda.is_available()):
    print('ERROR: cuda is not available, try running on CPU')
    sys.exit(1)

stylize(args)
