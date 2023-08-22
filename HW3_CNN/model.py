import torch
from torch import nn

class BasicBlock_18_34(nn.Module):
    """
    ResNet-18、34的基本网络结构
    """
    expansion = 1
    def __init__(self, input_channel:int, output_channel:int, stride:int=1, downsample=None) -> None:
        """
        function: 初始化用到的层
        @param input_channel: 残差结构block的输入通道数
        @param output_channel: 整个block的输出通道数
        @param stride: block中的第一层卷积层的stride
        @param downsample: 传入残差部分的网络
        """
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
    
    def forward(self, x):
        sample = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if(self.downsample != None):
            sample = self.downsample(sample)
        x += sample
        x = self.relu(x)
        return x

class BasicBlock_50_101_152(nn.Module):
    """
    ResNet-50, 101, 152基本block
    """
    expansion = 4
    def __init__(self, input_channel:int, output_channel:int, stride:int=1, downsample=None) -> None:
        """
        function: Block使用到的层初始化
        @param input_channel: block的输入channel数
        @param output_channel: block的输出channel数
        @param stride: block中第二个卷积层的stride
        @param dowmsample: 传入残差部分的网络
        """
        super().__init__()
        self.downsample = downsample
        temp_channel = int(output_channel / self.expansion)
        # 降维卷积
        self.conv1 = nn.Conv2d(input_channel, temp_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(temp_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(temp_channel, temp_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        # 升维卷积
        self.conv3 = nn.Conv2d(temp_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        sample = x
        # 降维卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 特征卷积
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 升维卷积
        x = self.conv3(x)
        x = self.bn2(x)
        if(self.downsample != None):
            sample = self.downsample(sample)
        x += sample
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    """
    ResNet网络搭建
    """
    def __init__(self, tag:str, num_classes:int, include_top:bool=True) -> None:
        """
        function: 搭建ResNet网络
        @param tag: "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "ResNet-152"
        @param num_classes: 分类数量
        @param include_top: 是否包含输出层
        """
        super().__init__()
        num_classes = int(num_classes)
        assert(num_classes > 0)
        assert(tag == "ResNet-18" or tag == "ResNet-34" or tag == "ResNet-50" or tag == "ResNet-101" or tag == "ResNet-152")
        assert(include_top == True or include_top == False)
        self.include_top = include_top
        if(tag == "ResNet-18"):
            block_num = [2, 2, 2, 2]
            basicblock = BasicBlock_18_34
        elif(tag == "ResNet-34"):
            block_num = [3, 4, 6, 3]
            basicblock = BasicBlock_18_34
        elif(tag == "ResNet-50"):
            block_num = [3, 4, 6, 3]
            basicblock = BasicBlock_50_101_152
        elif(tag == "ResNet-101"):
            block_num = [3, 4, 23, 3]
            basicblock = BasicBlock_50_101_152
        elif(tag == "ResNet-152"):
            block_num = [3, 8, 36, 3]
            basicblock = BasicBlock_50_101_152
        # input layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # hidden layers
        self.layer1 = self.make_layer(basicblock, block_num[0], 64, 64 * basicblock.expansion)
        self.layer2 = self.make_layer(basicblock, block_num[1], 64 * basicblock.expansion, 128 * basicblock.expansion)
        self.layer3 = self.make_layer(basicblock, block_num[2], 128 * basicblock.expansion, 256 * basicblock.expansion)
        self.layer4 = self.make_layer(basicblock, block_num[3], 256 * basicblock.expansion, 512 * basicblock.expansion)
        # output layers
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(512 * basicblock.expansion, num_classes)

        # initialize convertion layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
        
        return x

    def make_layer(self, basicblock, block_num, input_channel, output_channel) -> nn.Sequential:
        """
        function: 构建连续残差结构中的一个大block
        @param basicblock: 定义的basicblock
        @param block_num: 这个block中basicblock的数量
        @param input_channel: 整个block的输入channel
        @param output_channel: 整个block的输出channel
        @return: 构建完成的block
        """
        layer = []
        downsample = None
        stride = 1
        if(input_channel != output_channel):
            if(output_channel == input_channel * basicblock.expansion):
                downsample = nn.Sequential(
                    nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(output_channel)
                )
                stride = 1
            elif(output_channel != input_channel * basicblock.expansion):
                downsample = nn.Sequential(
                    nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(output_channel)
                )
                stride = 2
        layer.append(basicblock(input_channel, output_channel, stride, downsample))

        for i in range(1, block_num):
            layer.append(basicblock(output_channel, output_channel))
        
        return nn.Sequential(*layer)

def main():
    model = ResNet("ResNet-50", 11, True)
    print(model)
    return

if __name__ == "__main__":
    main()