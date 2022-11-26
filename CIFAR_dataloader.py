from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageFilter
"""
From the paper, the author suggest to use several transform:
Crop and Resized
Flip
Color Distort.(drop) --> Gray Scale
Color Distort.(jitter):
    Paper Suggest:


"""

class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def transform_apply(s = 1.0, training = False):
    if training:
        color_jit = transforms.ColorJitter(brightness= 0.8*s, contrast = 0.8*s, saturation = 0.8*s, hue = 0.2*s)
        random_color_jit = transforms.RandomApply([color_jit], p = 0.8)
        random_gray = transforms.RandomGrayscale(p = 0.2)
        color_distort = transforms.Compose([
                                            transforms.RandomResizedCrop(32),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            random_color_jit,
                                            random_gray,
                                            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        return color_distort
    else:
        return transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

class CIFAR_Dataloader(CIFAR10):
    """
    CIFAR10 has been saved in torchvision.datasets
    No need to be download
    Thanks to CIFAR teams for providing images
    Image size: 32*32*3. Where 3 is the 3 channel
    Just a little change from CIFAR10 pytorch source code. Partially Reference from here:
    https://pytorch.org/vision/main/_modules/torchvision/datasets/cifar.html
    """

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img) # convert to the PIL image for transform
        if self.transform is not None:
            f_1 = self.transform(img)
            f_2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return f_1, f_2, target

    def show_img(self, img_num):
        
        image1, image2, label = self.__getitem__(img_num)
        
        plt.subplot(1,2,1)
        plt.title(label)
        plt.imshow(image1.permute(1, 2, 0))
        plt.subplot(1,2,2)
        plt.imshow(image2.permute(1, 2, 0))
        plt.title(label)
        plt.show()

if __name__ == "__main__":
    """
    This is just for testing the code is working.
    Please comment it before you train or test the model
    """
    transform = transform_apply(s = 0.5, training = True)
    cifar = CIFAR_Dataloader(root='./data', train = True, transform= transform, download=True)
    x = cifar.__getitem__(0)
    print(x)
    cifar.show_img(0)

#     transform = transform_apply(s = 1.0, training = False)
#     cifar2 = CIFAR_Dataloader(root='./data', train = True, transform = transform, download=True)
#     x = cifar2.__getitem__(0)
#     cifar2.show_img(0)