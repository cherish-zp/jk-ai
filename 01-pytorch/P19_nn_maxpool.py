import torch
from torch import nn
from torch.nn import MaxPool2d
"""
最大池化： 
例如 ： 5*5 的图片 池化后变成 2*2 或者 1 的
"""





input = torch.tensor([
    [1,2,0,3,1],
    [3,4,5,6,7],
    [1,2,3,4,3],
    [2,3,4,5,6],
    [2,3,5,6,6],
    [1, 2, 0, 3, 1],
    [3, 4, 5, 6, 7],
    [1, 2, 3, 4, 3],
    [2, 3, 4, 5, 6],
    [2, 3, 5, 6, 6]
] , dtype=torch.float32)


print(input.shape)
"""
(-1 , 1, 5, 5)
(N, C, H, W)
N = batch size
C = 通道数
H = 高
W = 宽
"""
input = torch.reshape(input, (-1 , 1, 5, 5))
print(input.shape)

class TuDou(nn.Module):
    def __init__(self):
        super(TuDou, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3 , ceil_mode=True);

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudou = TuDou()
output = tudou(input)
print(output)


