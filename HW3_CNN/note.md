# HW3 CNN

## Dataset

直接使用opencv读入图片数据，在getitem下使用transform进行data augmentation

## Model

搭建ResNet-34作为model

## Train

交叉熵损失函数

Adam优化器

余弦退火调整学习率