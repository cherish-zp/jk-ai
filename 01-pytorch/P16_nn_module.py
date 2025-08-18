import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
# 用 tudui(x) 会自动执行 forward 方法。
output = tudui(x)
print(output)