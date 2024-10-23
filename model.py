import torch
import torch.nn as nn
import torchvision.models as models


class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        # 继承nn.Module
        super(SkinLesionClassifier, self).__init__()
        # self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 设置骨干网络，应用ResNet作为骨干网络，加载预训练的权重
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 将ResNet-50 的最后一层全连接层替换为一个新的线性层， 输入为原始全连接层输入的特征数
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # 额外的线性层分类器，良性或恶性
        self.malignant_classifier = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        # 全局平均池化层
        features = self.backbone.avgpool(
            # 残差模块层
            self.backbone.layer4(
                self.backbone.layer3(
                    self.backbone.layer2(
                        self.backbone.layer1(
                            # 最大池化层
                            self.backbone.maxpool(
                                # ReLu激活函数层
                                self.backbone.relu(
                                    # 批量归一化层
                                    self.backbone.bn1(
                                        # 第一个卷积层
                                        self.backbone.conv1(x))))))))).flatten(1)  # 将特征展平
        # 使用sigmoid将输出限制在[0, 1]范围内，输出为7个类别的概率
        diagnosis_output = torch.sigmoid(self.backbone.fc(features))
        # 将特征输入到是否为恶性的分类器，同样使用sigmod
        malignant_output = torch.sigmoid(self.malignant_classifier(features))
        return diagnosis_output, malignant_output
