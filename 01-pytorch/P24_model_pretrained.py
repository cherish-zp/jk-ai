import torchvision
from torch import nn

#train_data = torchvision.datasets.ImageNet(
#    root='../dataset',
#    split='train',
#    download=True,
#    transform=torchvision.transforms.ToTensor()
#)

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)


vgg16_true.add_module("add_linear", nn.Linear(1000, 10))

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))


vgg16_false.classifier[6] = nn.Linear(1000, 10)



