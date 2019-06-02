import torch
import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    def __init__(self, vgg_path='models/vgg16-00b39a1b.pth'):
        super(VGG16, self).__init__()

        # Load VGG16 skeleton, pretrained
        vgg16_features = models.vgg(pretrained=False)
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16_features.features

        # Turn off requires_grad
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2',
                  '15': 'relu3_3', '22': 'relu4_3'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if (name in layers):
                features[layers[name]] = x
                if(name == '22'):
                    break

        return(features)
