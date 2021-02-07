# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 08:15:36 2021

@author: Pierpaolo Sepe
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        vgg = models.vgg19(pretrained=True)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.model = vgg.features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)

        return features


def image_loader(path):
    imsize = 512 if torch.cuda.is_available() else 128
    image = Image.open(path)
    loader = transforms.Compose([transforms.Resize((imsize, imsize)),
                                 transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(input):
    batch_size, n_feture_maps, height, width = input.size()
    features = input.view(batch_size * n_feture_maps, height * width)

    G = torch.mm(features, features.t())
    batch_size, n_feture_maps, height, width = input.size()
    return G.div(batch_size * n_feture_maps * height * width)


def compute_content_loss(gen_feat, content_feat):
    content_loss = F.mse_loss(gen_feat, content_feat)
    return content_loss


def compute_style_loss(gen_feat, style_feat):
    G = gram_matrix(gen_feat)
    A = gram_matrix(style_feat)
    style_loss = F.mse_loss(G, A)
    return style_loss


def compute_loss(gen_features, content_features, style_features):
    style_loss = content_loss = 0
    for gen, cont, style in zip(gen_features, content_features, style_features):
        content_loss += compute_content_loss(gen, cont)
        style_loss += compute_style_loss(gen, style)

    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


device = torch.device("cuda")
content_image = image_loader('Images/painted_ladies.jpg')
style_image = image_loader('Images/starry_night.jpg')

generated_image = content_image.clone().requires_grad_(True)

model = VGG().to(device).eval()

n_epochs = 1000
content_weight = 0.1
style_weight = 10000000
optimizer = optim.LBFGS([generated_image])

for epoch in range(n_epochs):
    def closure():
        generated_image.data.clamp_(0, 1)
        gen_features = model(generated_image)
        content_features = model(content_image)
        style_features = model(style_image)
        total_loss = compute_loss(gen_features, content_features, style_features)
        optimizer.zero_grad()
        total_loss.backward()
        if epoch % 10 == 0:
            print(total_loss)
            imshow(generated_image)
            save_image(generated_image, "Result/generated.png")
        return total_loss.item()


    optimizer.step(closure)


