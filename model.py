from torch import nn

class CovidRegressionModel(nn.Module):
    def __init__(self, input_dim) -> None:
        """
        @param input_dim: 模型输入数据维度
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)