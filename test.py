import torch
from utils import setup_logger
import logging
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from dataset import SkinLesionDataset, get_transforms
from config import *

import warnings

# 忽略警告，减少输出的杂乱度，使输出结果更整洁
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def test_model(device, model_dir, output_dir):
    """
    用于测试模型的函数。

    参数:
    - device: 用于测试的设备 (CPU 或 GPU)。
    - model_dir: 模型文件所在的目录。
    - output_dir: 测试输出目录。
    """
    log_file = setup_logger('test', output_dir)

    # 准备测试数据集， 不使用病变分组
    test_dataset = SkinLesionDataset(
        ground_truth_csv=f'{dataset_dir}/2018/ISIC2018_Task3_Test_GroundTruth.csv',
        img_dir=f'{dataset_dir}/2018/ISIC2018_Task3_Test_Input',
        # 不使用数据增强
        transform=get_transforms(is_train=False)
    )

    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 获取最新的模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    # 如果不存在模型，则抛出异常
    if not model_files:
        raise FileNotFoundError("No model file found in the models directory.")
    # 找到模型目录中最新的模型文件
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
    # 构建完整的模型路径
    model_path = os.path.join(model_dir, latest_model)

    # 加载模型
    model = torch.load(model_path)
    # 将模型加载到设备
    model.to(device)
    # 转为评估模式
    model.eval()
    # 记录log
    logging.info(f"Model loaded from {model_path} and set to evaluation mode.")

    # 评估模型性能，真实值
    all_labels = []
    # 预测值
    all_preds = []

    # 不使用梯度
    with torch.no_grad():
        # 遍历测试集加载器
        for i, (images, diagnosis_labels, _) in enumerate(test_loader):
            # 将图片加载到设备
            images = images.to(device)
            # 将多分类标签加载到设备
            diagnosis_labels = diagnosis_labels.to(device)
            # 存储输出
            diagnosis_output, _ = model(images)
            # 将阈值设置为0.5，概率，如果大于0.5则标记为1，否则为0. 比如[1,0,0,0,0,0,0]
            diagnosis_preds = (diagnosis_output > 0.5).float()
            # 存储真实值
            all_labels.extend(diagnosis_labels.cpu().numpy())
            # 存储预测值
            all_preds.extend(diagnosis_preds.cpu().numpy())

            # 记录每个批次的预测结果
            for j in range(len(diagnosis_labels)):
                true_label = diagnosis_labels[j].cpu().numpy()
                pred_label = diagnosis_preds[j].cpu().numpy()
                logging.info(f"Sample {i * BATCH_SIZE + j + 1}: True: {true_label}, Predicted: {pred_label}")

    # 将真实值和预测值按照总共的类数量转为二维数组
    all_labels = np.array(all_labels).reshape(-1, NUM_CLASSES)
    all_preds = np.array(all_preds).reshape(-1, NUM_CLASSES)

    # 计算分类报告，精确率，召回率，F1 Score, 避免除以0的错误
    report = classification_report(all_labels, all_preds,
                                   target_names=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
                                   zero_division=0)
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    # 将信息记录在log中
    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Classification Report:\n{report}")

    # 打印测试准确率和报告
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Classification Report:\n{report}")

    # 标记测试完成
    logging.info("Testing completed. Results have been logged.")


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置测试目录和输出目录
    output_dir, model_dir = setup_test_dirs()

    # 调用测试函数
    test_model(device, model_dir, output_dir)
