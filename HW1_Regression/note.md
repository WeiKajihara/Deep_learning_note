# Process

## 1.csv数据读取

导入csv包使用reader()函数直接读取

```python
with open("hw1_regression\covid.train_new.csv", 'r') as f:
    csv_data = list(csv.reader(f))  # 读取所有数据并转成list
    feature_tags = csv_data[0]
    data = np.array(csv_data[1:])[:, 1:].astype(float) # 除id跟tags外的所有数据生成矩阵

    # 拆分features 与 targets
    features = data[:, :-1]
    targets = data[:, -1]
```

## 2.筛选特征

```python
	# 选择特征
    print(features.shape)
    # 选取得分前k个特征
    feature_dim = 16
    features_filter = SelectKBest(f_regression, k=feature_dim)
    features_new = features_filter.fit_transform(features, targets)
    print(features_new.shape)
    score = features_filter.scores_
    indices = np.argsort(score)[::-1]   # 得到得分前k的特征下标
    k_best_features = [feature_tags[i + 1] for i in indices[0:feature_dim]]
    print(f"Choose {feature_dim} features, features are {k_best_features}")
```

## 3.构建数据集

```python
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
                self.y = torch.tensor(data[indices, -1])
            elif(mode == "valid"):
                indices = [i for i in range(len(features_best)) if i % 5 == 0]
                self.y = torch.tensor(data[indices, -1])

            self.data = torch.tensor(features_best[indices])
            self.mode = mode
            self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / self.data.std(dim=0, keepdim=True)  # standardize
            assert feature_dim == self.data.shape[1]

            print(f"Finished loading the {mode} set of data, {len(self.data)} samples founded, each dim={feature_dim}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.y[index]
```

## 4.创建模型

```python
from torch import nn

class CovidRegressionModel(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)
```

## 5.组合训练