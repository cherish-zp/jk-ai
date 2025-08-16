import torch
import torchvision
from  torch import nn
from torch.nn import Linear
from  torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),download=True)


dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class TuDou(nn.Module):
    def __init__(self):
        super(TuDou, self).__init__()
        # torch.Size([64, 3, 32, 32])  3072 是单个样本的长度 3*32*32  , 【理解是 3072 的分类任务，分解成了10的分类任务】
        # 其实就是网络计算 output = input @ weight.T + bias   output 10 是结果， input 是输入 3072 ， weight 和bias 是可训练参数，在变换，讲 3072高纬度映射到 10 上
        self.linear1 = Linear(3*32*32, 10)   # 定义全连接层
    def forward(self, input):
        output = self.linear1(input)
        return output

tudou = TuDou()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])

    # 展平每个样本，保留批次维度 这里需要保留批次维度 加上 start_dim=1：获取到值是torch.Size([64, 3072]) ， 要不让就是 获取到单个样本长度是torch.Size([196608]) 这样单个样本维度就不对了
    output = torch.flatten(imgs, start_dim=1)  # 形状: [64, 3072]
    print(output.shape)

    output = tudou(output)  # 形状: [64, 10]
    print(output.shape)



