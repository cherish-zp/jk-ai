import torch
import torch.nn.functional as F

"""
conv 卷积神经网络 CNN, Convolutional Neural Network）
卷积操作
"""

## 输入层
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

#  shape 转换
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)



output = F.conv2d(input, kernel)
print(output.shape)
print(output)

