from csv import reader
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import torch
from torch import nn, optim
from model import CovidRegressionModel
from torch.utils.tensorboard import SummaryWriter

def get_feature_importance(feature_data, target_data, feature_dim = 8, feature_tags = None):
    """
    function: 选取输入feature_data中的得分前feature_dim个特征并返回
    ----------------------------------------------------------
    @param feature_data: 特征数据，type=ndarray
    @param target_data: 目标数据，type=ndarray
    @param feature_dim: 选取特征个数，type=int
    @param feature_tags: 特征名称，默认None，type=list

    return: 返回提取完后的feature_data(type=ndarray)与原feature中的下标list
    """
    feature_filter = SelectKBest(f_regression, k=feature_dim)
    feature_new = feature_filter.fit_transform(feature_data, target_data)
    score = feature_filter.scores_
    indices = np.argsort(score)[::-1]
    if(feature_tags != None):
        k_best_features = [feature_tags[i] for i in indices[0:feature_dim]]
        print(f"Have chosen {feature_dim} features, including {k_best_features}")
    return feature_new, indices[0:feature_dim]


# 构建数据集
class covidDataset(Dataset):
    def __init__(self, path, mode = "train", feature_dim = 16) -> None:
        """
        @param path: 数据路径
        @param mode: 读取数据模式，"train"读取训练集，"valid"读取验证集
        @param feature_dim: 数据筛选维度，筛选后的数据维度
        """
        super().__init__()
        with open(path, 'r') as f:
            csv_data = list(reader(f))  # read all data
            feature_tags = csv_data[0]
            feature_tags = feature_tags[1:] # find the feature tags
            data = np.array(csv_data[1:])[:, 1:].astype(float)  # create all feature matrix

            # divide features and targets
            features = data[:, :-1]
            targets = data[:, -1]

            # select best features
            features_best, col_indices = get_feature_importance(features, targets, feature_dim, feature_tags)

            # according to mode to judge which data is selected
            assert (mode == "train" or mode == "valid"), '(ERROR)Dataset mode error, only can be inputed "train" or "valid"'
            if(mode == "train"):
                indices = [i for i in range(len(features_best)) if i % 5 != 0]
                self.y = torch.tensor(data[indices, -1], dtype=torch.float32)
            elif(mode == "valid"):
                indices = [i for i in range(len(features_best)) if i % 5 == 0]
                self.y = torch.tensor(data[indices, -1], dtype=torch.float32)

            self.data = torch.tensor(features_best[indices], dtype=torch.float32)
            self.mode = mode
            self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / self.data.std(dim=0, keepdim=True)  # standardize
            assert feature_dim == self.data.shape[1]

            print(f"Finished loading the {mode} set of data, {len(self.data)} samples founded, each dim={feature_dim}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.y[index]

def main():
    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting parameters
    batch_size = 64
    epochs = 200
    feature_dim = 16
    learning_rate = 0.001

    # load data
    train_set = covidDataset("hw1_regression/covid.train_new.csv", "train", feature_dim)
    valid_set = covidDataset("hw1_regression/covid.train_new.csv", "valid", feature_dim)
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_set, batch_size, shuffle=False, drop_last=False)

    # create model
    model = CovidRegressionModel(feature_dim)
    model = model.to(device)

    # create loss funciton
    loss_func = nn.MSELoss()
    loss_func = loss_func.to(device)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # start tensorboard
    writer = SummaryWriter("hw1_regression/logs")

    # training
    min_valid_loss = 0.0
    min_epoch = 0
    for epoch in range(epochs):
        print(f"--------------------- Train Epoch: {epoch} start ---------------------")
        model.train()
        train_loss = 0.0
        valid_loss = 0.0
        train_step = 0
        valid_step = 0
        for data in train_dataloader:
            features, targets = data
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            optimizer.zero_grad()
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_step += 1

            print(f"Train step: {train_step} , Train loss:{loss.item()}")
        print(f"--------------------- Train Epoch: {epoch} finished ---------------------")
        print(f"Total train loss: {train_loss}")
        writer.add_scalar("Train loss", train_loss, epoch)

        # valid
        print(f"--------------------- Valid Epoch: {epoch} start ---------------------")
        model.eval()
        with torch.no_grad():
            for data in valid_dataloader:
                features, targets = data
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = loss_func(outputs, targets)
                
                valid_loss += loss.item()
                valid_step += 1

                print(f"Valid step: {valid_step} , Valid loss:{loss.item()}")
        print(f"--------------------- Valid Epoch: {epoch} finished ---------------------")
        print(f"Total valid loss: {valid_loss}")
        writer.add_scalar("Valid loss", valid_loss, epoch)

        # save model
        if(epoch == 0):
            min_valid_loss = valid_loss
            min_epoch = 0
            torch.save(model.state_dict(), "hw1_regression/model/CovidRegressionModel.pkl")
            print("The best model has been saved!")
        else:
            if(valid_loss < min_valid_loss):
                min_valid_loss = valid_loss
                min_epoch = epoch
                torch.save(model.state_dict(), "hw1_regression/model/CovidRegressionModel.pkl")
                print("The best model has been saved!")

    writer.close()
    print("Training finished!")
    print(f"The best model epoch is {min_epoch}, the best model has been saved!")
    return

def test():
    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting parameters
    batch_size = 64
    epochs = 20
    feature_dim = 16
    learning_rate = 0.001

    # load data
    train_set = covidDataset("hw1_regression/covid.train_new.csv", "train", feature_dim)
    valid_set = covidDataset("hw1_regression/covid.train_new.csv", "valid", feature_dim)
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_set, batch_size, shuffle=False, drop_last=False)

    for data in train_dataloader:
        features, targets = data
        print(features.shape)
    return

if __name__ == "__main__":
    main()
    # test()