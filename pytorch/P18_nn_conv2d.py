
import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False,  transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True , num_workers=0 , drop_last=True)

# 定义一个神经网络
class TuDou(nn.Module):
    def __init__(self):
        super(TuDou, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        return x

tuDou = TuDou()
writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    inputsImages, labels = data
    output =  tuDou(inputsImages)

    print(inputsImages.shape)
    print(output.shape)
    output = torch.reshape(output, (-1,3,30,30))
    writer.add_images('outputs', inputsImages, step)
    writer.add_images('outputs2', output, step)

    step += 1

writer.close()




