import os
import torch
from  CIFAR_dataloader import transform_apply
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from Decoder import downstream
from tqdm import tqdm
batch_size = 200
class evaluation:
    def __init__(self, top_N_acc, encoder_type = "ResNet18"):
        self.top_N_acc = top_N_acc
        self.encoder_type = encoder_type
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if encoder_type == "ResNet18":
            # you can change the path to eval other model. In this version, I haven't train others resnet only resnet18. You can train other model and eval
            path = './unsupervised_model/Upstream_model_with_resnet_ResNet18.pth'
            self.model = downstream(path = path, input_channel = 512, number_class = 10).to(self.device)
            path = './supervised_model/Downstream_model_with_resnet_ResNet18.pth'
            self.model.load_state_dict(torch.load(path))
    
    def evaluation(self):
        self.model.eval()
        transform = transform_apply()
        eval_=CIFAR10(root='./data', train=False, transform=transform, download=True)
        eval_data = DataLoader(eval_, batch_size = batch_size, shuffle = False, drop_last = True)
        top_k_dict = dict()
        for n in range(1, self.top_N_acc + 1):
            key = 'top'+str(n)
            top_k_dict[key] = 0
        # eval no need backpropagation
        total_images = 0
        with torch.no_grad():
            print("############################ Author: Jiaxuan Li, z5086369, UNSW ##############################")
            print("##############################################################################################")
            print("################################# ... Model Evaluation ...  ##################################")
            print("##############################################################################################")
            for batch, (images, labels) in tqdm(enumerate(eval_data)):
                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.model(images)

                total_images += images.size(0) # batch size
                prediction = torch.argsort(pred, dim = -1, descending = True)
                for n in range(1, self.top_N_acc + 1):
                    key = 'top'+str(n)
                    top_k_dict[key] += torch.sum((prediction[:, 0:n] == labels.unsqueeze(dim = -1)).any(dim = -1).float()).item()
        print("----------------- ResNet18 SimCLR Evaluation -----------------")
        for key in top_k_dict.keys():
            acc = (top_k_dict[key]/10000)*100
            print(f"                   {key} accuracy is  {acc:{2}.{4}}%")

if __name__ == "__main__":
    eval = evaluation(top_N_acc = 5, encoder_type = "ResNet18")
    eval.evaluation()