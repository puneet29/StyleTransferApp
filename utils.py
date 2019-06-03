import torch
from PIL import Image


def gram_matrix(tensor):
    (b, ch, h, w) = tensor.size
    features = tensor.view(b, ch, h*w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch*h*w)
    return(gram)


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if(size is not None):
        img = img.resize((size, size), Image.ANTIALIAS)
    if(scale is not None):
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    return(img)
