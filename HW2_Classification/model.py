from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        return self.model(x)
    
class LibriClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_dim, classify_num=41) -> None:
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, classify_num)
        )
    
    def forward(self, x):
        return self.model(x)

def main():
    model = LibriClassifier(39, 4, 64, 41)
    print(model)
    return

if __name__ == "__main__":
    main()