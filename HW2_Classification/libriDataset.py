import torch, os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

def metrix_move(metrix:torch.Tensor, step:int, dir:str="down") -> torch.Tensor:
    """
    function: 二维metrix向上或向下移动step个单位，上方或下方空出来的地方用第一行或最后一行重复填充
    ------------------------------------------------------------------
    @param metrix: 需要移动的二维Tensor矩阵
    @param step: 移动的单位
    @param dir: 移动方向，"up" or "down"
    @return: 返回移动后的矩阵
    """

    assert (dir == "down" or dir == "up")
    step = int(step)

    if dir == "down":
        left = metrix[0].repeat(step, 1)
        right = metrix[:-step]
    elif dir == "up":
        left = metrix[step:]
        right = metrix[-1].repeat(step, 1)
    
    return torch.cat((left, right), dim=0)

def frame_connect(metrix:torch.Tensor, connect_num:int) -> torch.Tensor:
    """
    funciton: 将二维metrix每一行前后接续(connect_num/2)个frame(帧), metrix的每一行都是一帧, 前后共接续(connect_num-1)个帧，合并后每一行有connect_num个帧
    ------------------------------------------------------------------
    @param metrix: tensor类型，需要接续的二维矩阵
    @param connect_num: 接续后的frame长度
    @return: 返回接续后的二维矩阵
    """

    # 接续思路：
    # 整个矩阵重复connect_num次，中间矩阵右侧依次上升一行，最后一行重复，左侧依次下降一行，第一行重复
    # 然后把对应行进行连接组合成一个新的二维矩阵
    connect_num = int(connect_num)
    assert (connect_num % 2 == 1 and connect_num > 0)
    if connect_num < 2:
        return metrix
    frame_num, feature_dim = metrix.size(0), metrix.size(1)
    metrix = metrix.repeat(1, connect_num).view(frame_num, connect_num, feature_dim).permute(1, 0, 2)   # metrix升维并重复connect_num次
    mid = int(connect_num / 2)   # 确定中心矩阵下标
    # 中心矩阵左右矩阵分别上下移动
    for i in range(1, mid+1):
        metrix[mid + i] = metrix_move(metrix[mid + i], i, "up")
        metrix[mid - i] = metrix_move(metrix[mid - i], i, "down")
    # 重连并降维
    metrix = metrix.permute(1, 0, 2).view(frame_num, feature_dim * connect_num)
    return metrix

# 构建数据集
class libriDataset(Dataset):
    def __init__(self, root_path, mode:str="train", frame_num:int=1, classify_num = 41) -> None:
        """
        function: 初始化时进行数据预处理
        ------------------------------------------------------------------
        @param root_path: 数据包根目录
        @param mode: "train" or "valid"，选择加载的数据为训练集或测试集
        @param frame_num: 连接后的frame个数，为奇数且大于0
        @param classify_num: 分类类别数量
        """
        super().__init__()
        assert (mode == "train" or mode =="valid")
        assert (frame_num > 0 and frame_num % 2 == 1)
        frame_num = int(frame_num)

        train_label_path = os.path.join(root_path, "train_labels.txt")

        # 获取训练文件名列表与targets字典
        train_filename_list = []
        train_targets_dict = {}
        with open(train_label_path, 'r') as f:
            all_data = f.readlines()
        for i in range(len(all_data)):
            all_data[i] = all_data[i].strip('\n').split(' ')
            train_filename_list.append(all_data[i][0])
            train_targets_dict[train_filename_list[i]] = np.array(all_data[i][1:]).astype(float)

        # 加载所有训练文件
        self.features = []
        self.targets = []
        for filename in train_filename_list:
            train_filepath = os.path.join(root_path, "feat", "train", f"{filename}.pt")
            data = torch.load(train_filepath)
            data = frame_connect(data, frame_num)
            for i in range(data.size(0)):
                if(mode == "train"):
                    if(i % 100 != 0):
                        self.features.append(data[i])
                        self.targets.append(torch.tensor(train_targets_dict[filename][i], dtype=torch.long))
                elif(mode == "valid"):
                    if(i % 100 == 0):
                        self.features.append(data[i])
                        self.targets.append(torch.tensor(train_targets_dict[filename][i], dtype=torch.long))
        print(f"Finish loading the {mode} set of data, {len(self.features)} samples are founded, each sample's dimension is {self.features[0].size(0)}")
    
    def __getitem__(self, index):
        return self.features[index], self.targets[index]
    
    def __len__(self):
        return len(self.features)

def main():
    # train_set = libriDataset("hw2_classification/libriphone", mode="train", frame_num=3)
    valid_set = libriDataset("hw2_classification/libriphone", mode="valid", frame_num=5, classify_num=41)
    # print(len(valid_set))
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, drop_last=False)

    pbar = tqdm(valid_loader)

    for _, data in enumerate(pbar):
        features, targets = data
        print(features.shape)
        print(targets.shape)

    # 获取训练文件名列表与targets字典
    """ train_filename_list = []
    train_targets_dict = {}
    with open("hw2_classification/libriphone/train_labels.txt", 'r') as f:
        all_data = f.readlines()
    for i in range(len(all_data)):
        all_data[i] = all_data[i].strip('\n').split(' ')
        train_filename_list.append(all_data[i][0])
        train_targets_dict[train_filename_list[i]] = np.array(all_data[i][1:]).astype(float)

    print(torch.tensor(train_targets_dict[train_filename_list[0]][0])) """
    return

if __name__ == "__main__":
    main()