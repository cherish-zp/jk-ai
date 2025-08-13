
import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False,  transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True , num_workers=0 , drop_last=True)

img , target = test_data[0]
print(img)
print(target)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    print("epoch第{}轮 , 一共的轮次 : {}".format(epoch  , len(test_loader)) )
    for data in test_loader:
        imgs , targets = data
        print('第{}轮 ， 第{}批次,获取的数量:{}'.format(epoch  , step ,len(data[0])) )
        writer.add_images('Epoch:{}'.format(epoch) , imgs  , step )
        step = step + 1
writer.close()

