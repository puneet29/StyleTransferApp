# Importing the resources
import argparse
import sys

import cv2
import torch
from torchvision import transforms
from transformer import TransformNet
from utils import load_cam_image, show_cam_image


def webcam(args):

    # Select GPU if available
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Load transformer network
    print("Loading Transformer Network")
    net = TransformNet()
    net.load_state_dict(torch.load(args.model))
    net.to(device)
    print("Loaded the Transformer Network")

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, args.width)
    cam.set(4, args.height)

    # Save video
    if(args.save):
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        out = cv2.VideoWriter(args.output, fourcc, args.fps,
                              (args.width, args.height))

    # Main loop
    with torch.no_grad():
        while(True):
            # Get webcam input
            ret_val, img = cam.read()

            # Mirror
            img = cv2.flip(img, 1)

            # Free up any cuda cache
            torch.cuda.empty_cache()

            # Generate content frame
            content_tensor = load_cam_image(img)
            content_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            content_image = content_transform(content_tensor)
            content_image = content_image.unsqueeze(0).to(device)

            # Get stylized frame
            output = net(content_image).cpu()
            img2 = show_cam_image(output[0])

            # Save frame to file
            if(args.save):
                out.write(img2)

            # Show webcam
            cv2.imshow('Webcam', img2)
            if(cv2.waitKey(1) == 27):
                break

    # Free up memory
    cam.release()
    out.release()
    cv2.destroyAllWindows()


# Command line arguments
eval_arg_parser = argparse.ArgumentParser(
    description="parser for fast-neural-style-webcam")
eval_arg_parser.add_argument("--model", type=str, required=True,
                             help="saved model to be used for stylizing the image.")
eval_arg_parser.add_argument("--cuda", type=int, required=True,
                             help="set it to 1 for running on GPU, 0 for CPU")
eval_arg_parser.add_argument("--width", type=int, default=640,
                             help="set the width of the image captured by camera, default 640")
eval_arg_parser.add_argument("--height", type=int, default=360,
                             help="set the height of the image captured by camera, default 360")
eval_arg_parser.add_argument("--save", type=bool, default=False,
                             help="save captured frame to video, default False")
eval_arg_parser.add_argument("--output", type=str, default='output.avi',
                             help="path to save video, set --save to True, default 'output.avi'")
eval_arg_parser.add_argument("--codec", type=str, default='XVID',
                             help="codec of output video, default XVID")
eval_arg_parser.add_argument("--fps", type=int, default=20,
                             help="FPS of output video. Adjust according to your system, default 20")


args = eval_arg_parser.parse_args()

if(args.cuda and not torch.cuda.is_available()):
    print('ERROR: cuda is not available, try running on CPU')
    sys.exit(1)

webcam(args)
