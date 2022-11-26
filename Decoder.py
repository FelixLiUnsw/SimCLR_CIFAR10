import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50
from ProjectionHead import SimCLRUnsupervised18,SimCLRUnsupervised34, SimCLRUnsupervised50
class downstream(nn.Module):
    def __init__(self, path, input_channel = 512, number_class = 10) -> None:
        super(downstream, self).__init__()
        self.encoder = SimCLRUnsupervised18().resnet
        temp_dict = dict()
        for k, v in torch.load(path).items():
            if "projection_head" in k:
                continue
            else:
                temp_dict[k.replace('resnet.', '')] = v
        self.encoder.load_state_dict(temp_dict)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.decoder = nn.Sequential(
                        nn.Linear(input_channel, 256, bias = False),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace = True),
                        nn.Linear(256, number_class, bias = True)
                        )
    def forward(self, x):
        representation = self.encoder(x)
        representation = torch.flatten(representation, start_dim = 1)
        ret = self.decoder(representation)
        return ret


# The following code is just for checking the architecture of the model
# only decoder allow to gradient update
# encoder section are frozen.
# You can check if you want to 
# Make sure you common the following lines when you ran the downstream traning

if __name__ == '__main__':
    path = './unsupervised_result/100_epoch_200_batch_size/Upstream_model_with_resnet_ResNet18.pth'
    model = downstream(path = path, input_channel = 512, number_class = 10)
    for parameter in model.parameters():
        print(parameter)
