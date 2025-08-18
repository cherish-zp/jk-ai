from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image

class MyData(Dataset):
    def __init__(self,root_dir , label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.img_path[index]
        return img, label

    def __len__(self):
        return len(self.img_path)



if __name__=='__main__':
    root_dir = f"/Users/zhangpeng/code_bigmodel/jk-ai/00-data/hymenoptera_data/train"
    ants_dataset =  MyData(root_dir , label_dir='ants')
    bees_dataset =  MyData(root_dir , label_dir='bees')

    train_dataset = ants_dataset + bees_dataset

    print(len(train_dataset))




