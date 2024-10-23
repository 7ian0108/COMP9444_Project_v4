import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import SkinLesionClassifier
from dataset import SkinLesionDataset, get_transforms
from utils import setup_logger
from config import *

import logging
import os

import warnings

# 忽略特定类的警告
warnings.filterwarnings("ignore", category=UserWarning)


def train_model(device, output_dir, model_dir):
    log_file = setup_logger('train', output_dir)

    # 准备数据集
    train_dataset = SkinLesionDataset(
        ground_truth_csv=f'{dataset_dir}/2018/ISIC2018_Task3_Training_GroundTruth.csv',
        lesion_groupings_csv=f'{dataset_dir}/2018/ISIC2018_Task3_Training_LesionGroupings.csv',
        img_dir=f'{dataset_dir}/2018/ISIC2018_Task3_Training_Input',
        # 应用数据增强
        transform=get_transforms(is_train=True)
    )

    # 不使用分类组
    valid_dataset = SkinLesionDataset(
        ground_truth_csv=f'{dataset_dir}/2018/ISIC2018_Task3_Validation_GroundTruth.csv',
        img_dir=f'{dataset_dir}/2018/ISIC2018_Task3_Validation_Input',
        # 不使用数据增强
        transform=get_transforms(is_train=False)
    )

    # 根据weight的权重进行随机采样，用于处理类别不平衡的情况，允许样本被重复采样
    train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset), replacement=True)
    # 创建训练数据加载器，使用加权采样器处理类别分配不均衡的问题
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    # 创建验证数据加载器，不允许打乱顺序
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数和优化器
    # 调用模型
    model = SkinLesionClassifier(num_classes=NUM_CLASSES).to(device)
    # 使用二分类交叉熵损失
    criterion_diagnosis = nn.BCELoss()  # 多标签分类任务
    criterion_malignant = nn.BCELoss()  # 恶性/良性二分类任务
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    # 初始化变量为无穷大，表示验证集的最大损失
    best_val_loss = float('inf')
    # 早停机制的耐心计时器
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # 训练模型
        model.train()
        # 初始化多标签任务和二分类任务的损失（累积）
        running_train_diagnosis_loss = 0.0
        running_train_malignant_loss = 0.0

        # 显示进度条
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")
        # 遍历数据加载器 （图片信息，多标签，是否为恶性病变）
        for images, diagnosis_labels, is_malignant in train_progress:
            # 将数据移动到设备
            images = images.to(device)
            # 将标签移动到设备
            diagnosis_labels = diagnosis_labels.to(device)
            # 将分类移动到设备
            is_malignant = is_malignant.to(device).unsqueeze(1)

            # 将梯度设置为0
            optimizer.zero_grad()
            # 获取模型的输出
            diagnosis_output, malignant_output = model(images)

            # 获取多分类的损失
            loss_diagnosis = criterion_diagnosis(diagnosis_output, diagnosis_labels)
            # 获取二分类的损失
            loss_malignant = criterion_malignant(malignant_output, is_malignant)

            # 加和两种损失
            total_loss = loss_diagnosis + loss_malignant
            # 反向传播
            total_loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()

            # 累积多分类损失
            running_train_diagnosis_loss += loss_diagnosis.item()
            # 累积二分类损失
            running_train_malignant_loss += loss_malignant.item()

            # 在进度条动态更新损失
            train_progress.set_postfix({
                "Diagnosis Loss": f"{loss_diagnosis.item():.4f}",
                "Malignant Loss": f"{loss_malignant.item():.4f}"
            })

        # 计算多分类训练平均损失
        avg_train_diagnosis_loss = running_train_diagnosis_loss / len(train_loader)
        # 计算二分类训练平均损失
        avg_train_malignant_loss = running_train_malignant_loss / len(train_loader)

        # 将训练损失记录到log中
        logging.info(
            f"Epoch {epoch + 1}, Training Diagnosis Loss: {avg_train_diagnosis_loss:.4f}, Training Malignant Loss: {avg_train_malignant_loss:.4f}")

        # Validation 验证集
        # 设置为验证模式
        model.eval()
        # 初始化验证损失
        running_val_diagnosis_loss = 0.0

        # 显示验证过程的进度条
        valid_progress = tqdm(valid_loader, desc=f"Epoch {epoch + 1} [Validation]")
        # 禁用梯度计算
        with torch.no_grad():
            # 遍历验证数据加载器
            for images, diagnosis_labels, _ in valid_progress:
                # 图像加载到设备
                images = images.to(device)
                # 多分类标签加载到设备
                diagnosis_labels = diagnosis_labels.to(device)

                # 存储模型验证的输出
                diagnosis_output, _ = model(images)
                # 记录模型的损失
                loss_diagnosis = criterion_diagnosis(diagnosis_output, diagnosis_labels)
                # 累积损失
                running_val_diagnosis_loss += loss_diagnosis.item()
                # 在进度条动态更新损失
                valid_progress.set_postfix({
                    "Diagnosis Loss": f"{loss_diagnosis.item():.4f}"
                })
        # 计算平均损失
        avg_val_diagnosis_loss = running_val_diagnosis_loss / len(valid_loader)
        # 记录log
        logging.info(f"Epoch {epoch + 1}, Validation Diagnosis Loss: {avg_val_diagnosis_loss:.4f}")

        # Save model 保存模型
        # torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'best_model_epoch_{epoch + 1}.pth'))
        torch.save(model, os.path.join(model_dir, f'best_model_epoch_{epoch + 1}.pth'))

        # Early stopping 早停机制，如果下一步损失比上一次损失小，则继续训练
        if avg_val_diagnosis_loss < best_val_loss:
            best_val_loss = avg_val_diagnosis_loss
            patience_counter = 0
        # 否则，早停耐心值加一
        else:
            patience_counter += 1
        # 当早停耐心值达到特定的阈值之后，停止训练，在log中记录
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logging.info("Early stopping triggered!")
            break
    # 记录训练完成
    logging.info("Training completed.")


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置训练目录和日志
    OUTPUT_DIR, MODEL_DIR = setup_train_dirs()

    # 开始训练
    train_model(device, OUTPUT_DIR, MODEL_DIR)

