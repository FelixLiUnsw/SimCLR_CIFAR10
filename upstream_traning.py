import os
import torch
from NT_Loss import NT_Xent_Loss
from  CIFAR_dataloader import CIFAR_Dataloader, transform_apply
from ProjectionHead import SimCLRUnsupervised18, SimCLRUnsupervised34, SimCLRUnsupervised50
from torch.utils.data import DataLoader
from tqdm import tqdm

strength = 0.5
batch_size = 200
learning_rate = 1.0 # start, it will decrease by using scheduler



class unsupervised_training:
    def __init__(self, output_layer = 128, encoder_type = "ResNet18"):
        self.output_layer = output_layer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        self.encoder_type = encoder_type
        print("################################ Device: {} #################################".format(self.device))
        if encoder_type == "ResNet18":
            self.unsupervised_model = SimCLRUnsupervised18(self.output_layer).to(self.device)

        elif encoder_type == "ResNet34":
            self.unsupervised_model = SimCLRUnsupervised34(self.output_layer).to(self.device)
        
        elif encoder_type == "ResNet50":
            self.unsupervised_model = SimCLRUnsupervised50(self.output_layer).to(self.device)
        else:
            raise("You need to choose an encoder!")
        self.Loss = NT_Xent_Loss().to(self.device)

    def get_lr(self, optimizer):
      for param_group in optimizer.param_groups:
          return param_group['lr']

    def unsupervised_model_traning(self, epochs, optimizer = 'SGD'):
        print("############################ Author: Jiaxuan Li, z5086369, UNSW ##############################")
        print("##############################################################################################")
        print("#######################  Unsupervised Learning Traning In Process ...  #######################")
        print("##############################################################################################")
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.unsupervised_model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.001)
        loss = NT_Xent_Loss().to(self.device)
        self.unsupervised_model.train()
        transform = transform_apply(s = strength, training = True)
        CIFAR_traning = CIFAR_Dataloader(root='./data', train = True, transform= transform, download=True)
        training_data = DataLoader(CIFAR_traning, batch_size = batch_size, shuffle = True, drop_last = True)
        path_experiment = "./experimental_result"
        for epoch in tqdm(range(1, epochs + 1)):
            self.unsupervised_model.train()
            total_loss = 0
            num_of_batch = 0
            for batch, (image1, image2, label) in enumerate(training_data):
                image1, image2, _ = image1.to(self.device), image2.to(self.device), label.to(self.device)
                zi = self.unsupervised_model(image1)
                zj = self.unsupervised_model(image2)
                loss = self.Loss(zi, zj, temperature = 0.5)

                print("batch = ", batch, "len(image) = ", len(image1), " loss = ", loss)
                num_of_batch += 1
                optimizer.zero_grad()
                loss.backward()
                loss.requires_grad_()
                optimizer.step()
                total_loss += loss.detach().item()
            average_batch_loss = total_loss/num_of_batch
            print(f"Epoch {epoch} with learning rate {self.get_lr(optimizer)} ****** The average batch loss is : {average_batch_loss}")
            with open(os.path.join(path_experiment, "unsupervised_loss.txt"), "a") as f:
                f.write(str(epoch)+" " + str(average_batch_loss)+" "+str(self.get_lr(optimizer))+" "+str(total_loss/50000)+"\n")
            scheduler.step()
            if epoch % 10 == 0:
                unsupervised_path = "./unsupervised_model"
                path = os.path.join(unsupervised_path, f'Upstream_model_with_resnet_{self.encoder_type}.pth')
                torch.save(self.unsupervised_model.state_dict(), path)

if __name__ == "__main__":
    traning = unsupervised_training(output_layer = 128)
    traning.unsupervised_model_traning(200)