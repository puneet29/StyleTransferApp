import cv2
import torch
from torchvision import transforms


def gram_matrix(tensor):
    # needs testing
    _, d, h, w = tensor.shape
    tensor = tensor.view(d, h*w)

    gram = torch.mm(tensor, tensor.t())
    return(gram)


def load_image(path):
    return(cv2.imread(path))


def tensor2image(img, max_size=None):
    if(max_size == None):
        tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    else:
        if(max(img.size) > max_size):
            size = max_size
        else:
            size = max(img.size)
        tensor = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    tensor = tensor(img).unsqueeze(dim=0)
    return(tensor)
