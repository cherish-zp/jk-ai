import torchvision.datasets

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets


# 图片类型转化成ToTensor类型
dataset_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True,transform=dataset_transform , download=True )
test_set  = torchvision.datasets.CIFAR10(root='./dataset', train=False,transform=dataset_transform , download=True)

print(test_set[0])
img , label = test_set[0]
print(img)
print(label)

writer = SummaryWriter("logs")
for i in range(10):
    img , label  = train_set[i]
    writer.add_image("test_set" , img , i )

writer.close()