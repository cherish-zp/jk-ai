from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "/Users/zhangpeng/code_bigmodel/jk-ai/00-data/hymenoptera_data/train/ants/1924473702_daa9aacdbe.jpg" ;
img = Image.open(img_path)


writer = SummaryWriter("logs")

#1. transform 该如何被使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img" , tensor_img)
writer.close()
print(tensor_trans)
