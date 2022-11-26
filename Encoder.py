import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50

"""
In Resnet from torchvision, kernel size is quite big for 32*32 image in conv1.
The original kernel size is 7. Now, we will change it to 3. and change the stride to 1 and change padding to 1 (suitable for our task)
The encoder (resnet18, resnet34, resnet50) needs to retain the information as much as possible even though there are some information in the image is redundancy
The goal of the encoder is to do the feature extraction and representation extraction. The maximum pooling will remove some weak signals.
Therefore, we need to remove the maximum pooling layer as well.
Meanwhile, the resent is currently an encoder not a classifier. Hence, fully connection layer maybe not useful for us.
Hence, the fully connected layer need to be removed as well.
The remaining section is same.
Hence, we need to modify the resnet50
Here are some introduction about resnet50 from dataloaders
https://medium.com/@lucrece.shin/chapter-3-transfer-learning-with-resnet50-from-dataloaders-to-training-seed-of-thought-67aaf83155bc
"""

class resNet18(nn.Module):
    def __init__(self):
        super(resNet18,self).__init__()
        self.encoder = []
        for layer, module in resnet18().named_children():
            if layer == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size = 3,  padding=1, bias=False)
            if layer == 'fc' or layer == 'maxpool':
                self.encoder.append(nn.Identity())
            else:
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)
    def forward(self, x):
        x = self.encoder(x)
        return x

class resNet34(nn.Module):
    def __init__(self):
        super(resNet34,self).__init__()
        self.encoder = []
        for layer, module in resnet34().named_children():
            if layer == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            if layer == 'fc' or layer == 'maxpool':
                self.encoder.append(nn.Identity())
            else:
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)
    def forward(self, x):
        x = self.encoder(x)
        return x

class resNet50(nn.Module):
    def __init__(self):
        super(resNet50,self).__init__()
        self.encoder = []
        for layer, module in resnet50().named_children():
            if layer == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            if layer == 'fc' or layer == 'maxpool':
                self.encoder.append(nn.Identity())
            else:
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        return x

if __name__ == "__main__":
    """
    This main function is for checking the layer and channels, please comment it if you use the code
    """

    print("###### ResNet18 Latent Factor = 512  #######")
    ResNet18 = resNet18()
    print(ResNet18)

#     print("###### ResNet34 Latent Factor = 512  #######")
#     ResNet34 = resNet34()
#     print(ResNet34)

#     print("###### ResNet50 Latent Factor = 2048 #######")
#     ResNet50 = resNet50()
#     print(ResNet50)



