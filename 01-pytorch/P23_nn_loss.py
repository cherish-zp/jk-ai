"""
损失函数
"""

import torch
from torch.nn import L1Loss
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss

inputs = torch.tensor([1,2,3] , dtype=torch.float32)
outputs = torch.tensor([1,2,5] , dtype=torch.float32)

print(inputs.shape)
"""
reshape 操作：改变张量的形状（shape），但不改变其数据的总元素数量。

目标形状：(1, 1, 1, 3)，表示：

第1维（1）：批次大小（batch size）。

第2维（1）：通道数（channels）。

第3维（1）：高度（height）。

第4维（3）：宽度（width）。
"""
inputs = torch.reshape(inputs ,   (1,1,1,3 ));
outputs = torch.reshape(outputs , (1,1,1,3 ));

# L1Loss  差值 的和的平均值 （ |1-1| + |2-2| + |3-5| ） = 2
loss  = L1Loss(reduction='sum')
result = loss(inputs , outputs)

print(result)
        #差值的和的平均值 （ |1-1| + |2-2| + |3-5| ）/ 3
loss  = L1Loss(reduction='mean')
result = loss(inputs , outputs)
print(result)

# MSELoss #差值的绝对值的平方 的和的平均值 （ |1-1|² + |2-2|² + |3-5|² ）/ 3
loss  = MSELoss(reduction='mean')
result = loss(inputs , outputs)
print(result)

# 交叉熵
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))

loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)
