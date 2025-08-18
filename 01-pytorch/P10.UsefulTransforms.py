from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

'''
transforms 主要对图片进行处理
'''

writer = SummaryWriter("logs")

img = Image.open("/Users/zhangpeng/code_bigmodel/jk-ai/00-data/hymenoptera_data/123.png")
print(img)

# toTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(img_tensor)
writer.add_image("Tensor_img" , img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
img_norm = trans_norm(img_tensor)
print(img_tensor[0][0][0])

writer.add_image("Normalize" , img_norm , 2 )

# resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("resize" , img_resize , 0)
print(trans_resize)


# compose -  resize -2
trans_resize_2 = transforms.Resize(512)
#PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2 , trans_totensor])
img_resize_2 = trans_compose(img)

writer.add_image("resize" , img_resize_2 , 1)

writer.close()