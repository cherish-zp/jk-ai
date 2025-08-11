from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os

'''

pip install opencv-python
pip install tensorboard

使用方法： tensorboard --logdir=logs --port=6007 --bind_all
'''

if __name__ == '__main__':

    root_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-data/hymenoptera_data/"
    image_path = os.path.join(root_dir, "train/ants/0013035.jpg")
    imp_PIL = Image.open(image_path)
    img_array = np.array(imp_PIL)

    print(img_array.shape)
    print(type(img_array))

    writer = SummaryWriter("logs")

    writer.add_image("test" , img_array,  1  , dataformats="HWC")

    for i in range(0 , 100):
        writer.add_scalar("y=x" , i , i )
    writer . close()