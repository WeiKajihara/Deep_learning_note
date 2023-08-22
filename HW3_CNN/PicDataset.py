import os, torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

def load_pic(folder_path:str, pic_name:str) -> (np.ndarray, torch.Tensor):
    """
    function: 加载图片并进行预处理，获取标签，返回处理后的ndarray图像与Tensor标签
    -------------------------------------------------------------
    @param folder_path: 图像文件夹路径
    @param pic_name: 图像名称
    @param img_size: 图像与处理后的尺寸大小
    @return: (img, target)
    """
    img_path = os.path.join(folder_path, pic_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target = pic_name.split("_")[0]
    target = torch.tensor(int(target), dtype=torch.long)
    return img, target

# 构建数据集
class PicDataset(Dataset):
    def __init__(self, root_path, transform=None, mode:str="train") -> None:
        """
        function: 数据集初始化
        -------------------------------------------------------------
        @param root_path: 所有数据根目录
        @param mode: "train" or "valid", 选择加载的数据集
        """
        super().__init__()
        assert(mode == "train" or mode == "valid")
        self.transform = transform
        data_path = os.path.join(root_path, mode)
        img_list = os.listdir(data_path)
        self.imgs = []
        self.targets = []
        for img_name in img_list:
            img, target = load_pic(data_path, img_name)
            self.imgs.append(img)
            self.targets.append(target)

        print(f"Finish loading {mode} set of data, {len(self.imgs)} samples have been founded")

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        if self.transform != None:
            return self.transform(self.imgs[index]), self.targets[index]
        return self.imgs[index], self.targets[index]

def main():
    valid_set = PicDataset("hw3_CNN/hw3_data", "valid", (224, 224))
    loader = DataLoader(valid_set, 16, shuffle=False, drop_last=False)
    train_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomHorizontalFlip(), # 随机翻转图片
        transforms.RandomRotation(15.0), # 随机旋转图片
        transforms.ToTensor()
    ])
    for data in loader:
        imgs, targets = data
        imgs = train_transform(imgs)
        print(type(imgs))
        print(imgs.shape)
    # print(type(cv2.imread("hw3_CNN/hw3_data/train/0_0.jpg")))
    return

if __name__ == "__main__":
    main()