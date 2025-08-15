import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter
"""
最大池化： 
例如 ： 5*5 的图片 池化后变成 2*2 或者 1 的
"""

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                       download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=64 ,shuffle=True)

class TuDou(nn.Module):
    def __init__(self):
        super(TuDou, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3 , ceil_mode=True);

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tuDou = TuDou()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    inputsImages, labels = data
    output =  tuDou(inputsImages)

    print(inputsImages.shape)
    print(output.shape)
    # add_images 只能增加3通道的 所有进行转化
    output = torch.reshape(output, (-1,3,11,11))
    writer.add_images('-outputs', inputsImages, step)
    writer.add_images('-outputs2', output, step)

    step += 1

writer.close()
    
