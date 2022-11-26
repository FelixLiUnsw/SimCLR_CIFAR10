import os
import torch
from NT_Loss import NT_Xent_Loss
from  CIFAR_dataloader import CIFAR_Dataloader, transform_apply
from Decoder import downstream
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
from tqdm import tqdm


strength = 1.0
batch_size = 200
learning_rate = 0.1 # start, it will decrease by using scheduler

class downstream_traning:
    def __init__(self, input_layer, number_class = 10, encoder_type = "ResNet18"):
        self.input_layer = input_layer
        self.number_class = number_class
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder_type = encoder_type
        if encoder_type == "ResNet18":
            # YOU CAN MODIFY HERE TO TEST OTHER MODEL (Need a traning for resnet34 and resnet 50)
            path = './unsupervised_model/Upstream_model_with_resnet_ResNet18.pth'
            self.supervised_model = downstream(path = path, input_channel = self.input_layer, number_class = self.number_class).to(self.device)
    
    def get_lr(self, optimizer):
      for param_group in optimizer.param_groups:
          return param_group['lr']

    def traning(self, epochs, optimizer = 'SGD'):
        print("############################ Author: Jiaxuan Li, z5086369, UNSW ##############################")
        print("##############################################################################################")
        print("########################  Supervised Learning Traning In Process ...  ########################")
        print("##############################################################################################")
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.supervised_model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        transform = transform_apply(s = strength, training = True)
        CIFAR_traning_downstream = CIFAR10(root='./data', train = True, transform = transform, download=True)
        downstream_training_data = DataLoader(CIFAR_traning_downstream, batch_size = batch_size, shuffle = True, drop_last = True)
        Downstream_loss = nn.CrossEntropyLoss()
        path_experiment = "./experimental_result"
        for epoch in range(1, epochs+1):
            self.supervised_model.train()
            total_loss = 0
            num_of_batch = 0
            for batch, (images, label) in enumerate(downstream_training_data):
                
                images, label = images.to(self.device), label.to(self.device)
                pred_images = self.supervised_model(images)
                loss = Downstream_loss(pred_images, label)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_of_batch += 1
                total_loss += loss.detach().item()
            average_batch_loss = total_loss/num_of_batch
            print(f"Epoch {epoch} with learning rate {self.get_lr(optimizer)} ****** The average batch loss is : {average_batch_loss}")
            with open(os.path.join(path_experiment, "Supervised_loss.txt"), "a") as f:
                f.write(str(epoch)+" " + str(average_batch_loss)+" "+str(self.get_lr(optimizer))+" "+str(total_loss/50000)+"\n")
            scheduler.step()
            if epoch % 10 == 0:
                supervised_path = "./supervised_model"
                path = os.path.join(supervised_path, f'Downstream_model_with_resnet_{self.encoder_type}.pth')
                torch.save(self.supervised_model.state_dict(), path)

if __name__ == "__main__":
    traning = downstream_traning(input_layer = 512, number_class = 10, encoder_type = "ResNet18")
    traning.traning(200)