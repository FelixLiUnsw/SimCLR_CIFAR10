
import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import resNet18, resNet34, resNet50
"""
Adding the projection head on the encoder, the projection head is just a 2 Layer MLP
"""

class SimCLRUnsupervised18(nn.Module):
    """
    This section is same as the above image structure
    In projection head, we used 3-layer MLP rather than 2 layer MLP(paper indicates), because the input channel maybe to large, it has 2048
    To reduce the the case of "nonlinearity", we applied the 3 layers of MLP.
    """
    def __init__(self, projection_output = 128):
        super(SimCLRUnsupervised18, self).__init__()
        # encoder f(.)
        self.resnet = resNet18()
        # projection head g(.)
        self.projection_head = nn.Sequential(nn.Linear(512, 512, bias = False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(),
                               nn.Linear(512, projection_output, bias = False),
                               nn.BatchNorm1d(projection_output)
                              )
    def forward(self, x):
        h = self.resnet(x)
        h = torch.flatten(h, start_dim = 1)
        z = self.projection_head(h)
        #z = F.normalize(z, dim = -1)
        return z

class SimCLRUnsupervised34(nn.Module):
    """
    This section is same as the above image structure
    In projection head, we used 3-layer MLP rather than 2 layer MLP(paper indicates), because the input channel maybe to large, it has 2048
    To reduce the the case of "nonlinearity", we applied the 3 layers of MLP.
    """
    def __init__(self, projection_output = 128):
        super(SimCLRUnsupervised34, self).__init__()
        # encoder f(.)
        self.resnet = resNet34()
        # projection head g(.)
        self.projection_head = nn.Sequential(nn.Linear(512, 256, bias = False),
                               #nn.BatchNorm1d(256),
                               nn.ReLU(),
                               nn.Linear(256, projection_output, bias = False)
                              )
    def forward(self, x):
        h = self.resnet(x)
        h = torch.flatten(h, start_dim = 1)
        z = self.projection_head(h)
        z = F.normalize(z, dim = -1)
        return z

class SimCLRUnsupervised50(nn.Module):
    """
    This section is same as the above image structure
    In projection head, we used 3-layer MLP rather than 2 layer MLP(paper indicates), because the input channel maybe to large, it has 2048
    To reduce the the case of "nonlinearity", we applied the 3 layers of MLP.
    """
    def __init__(self, projection_output = 128):
        super(SimCLRUnsupervised50, self).__init__()
        # encoder f(.)
        self.resnet = resNet50()
        # projection head g(.)
        self.projection_head = nn.Sequential(nn.Linear(2048, 1024, bias = False),
                               nn.BatchNorm1d(1024),
                               nn.ReLU(),
                               nn.Linear(1024, 512, bias = False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(),
                               nn.Linear(512, projection_output, bias = False)
                              )
    def forward(self, x):
        h = self.resnet(x)
        h = torch.flatten(h, start_dim = 1)
        z = self.projection_head(h)
        z = F.normalize(z, dim = -1)
        return z

# if __name__ == "__main__":
#     """
#     This main function is for checking the layer and channels, please comment it if you use the code
#     """

#     print("###### ResNet18 + 2 Layers MLP | Latent Factor = 512 | projection latent factor = 128 #######")
#     unsup18 = SimCLRUnsupervised18()
#     print(unsup18)

#     print("###### ResNet34 + 2 Layers MLP | Latent Factor = 512 | projection latent factor = 128 #######")
#     unsup34 = SimCLRUnsupervised34()
#     print(unsup34)

#     print("###### ResNet50 + 3 Layers MLP | Latent Factor = 2048 | projection latent factor = 128 #######")
#     unsup50 = SimCLRUnsupervised50()
#     print(unsup50)