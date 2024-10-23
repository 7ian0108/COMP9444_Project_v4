import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class SkinLesionDataset(Dataset):
    def __init__(self, ground_truth_csv, img_dir, lesion_groupings_csv=None, transform=None):
        """
        Args:
            ground_truth_csv (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            lesion_groupings_csv (string, optional): Path to the CSV file with lesion groupings. Default is None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ground_truth_data = pd.read_csv(ground_truth_csv)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

        if lesion_groupings_csv:
            # If lesion_groupings_csv is provided, merge to get lesion_id information
            self.lesion_groupings_data = pd.read_csv(lesion_groupings_csv)
            # 将两个文件合并，左侧喂真实值，右侧为病变分组信息， 将逐渐设置为image，图像文件名
            self.data = pd.merge(self.ground_truth_data, self.lesion_groupings_data, on='image')
            self.lesion_groups = self._group_lesions()
            # 生成样本索引列表和对应的权重
            self.samples, self.weights = self._prepare_samples_and_weights()
            # 是否使用病变分组信息
            self.use_groups = True
        else:
            # Otherwise, just use the ground truth data directly
            self.data = self.ground_truth_data
            self.use_groups = False

        # 如果未指定 transform，则使用默认的 transform 将 PIL 图像转换为张量
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为张量 [C, H, W]  通道数量，图像高度， 图像宽度
            ])

    def _group_lesions(self):
        """ Group images by lesion_id and determine if they are benign or malignant. """
        lesion_groups = {}
        # 良性类别
        benign_labels = ['NV', 'BKL']
        # 恶性类别
        malignant_labels = ['MEL', 'BCC', 'AKIEC']

        # 遍历每一行，提取每一行的样本病变信息
        for _, row in self.data.iterrows():
            # 获取每一行的id
            lesion_id = row['lesion_id']
            # 如果在恶性类别中，标记为1， 则标记为恶性病变
            is_malignant = any(row[label] == 1 for label in malignant_labels)
            # 如果在良性类别中，标记为1，且恶性类别中没有标记为1， 则标记为良性病变
            is_benign = any(row[label] == 1 for label in benign_labels) and not is_malignant

            # 创建一个初始化的字典
            if lesion_id not in lesion_groups:
                lesion_groups[lesion_id] = {
                    'images': [],
                    'is_malignant': False,
                    'is_benign': False
                }
            # 添加文件名
            lesion_groups[lesion_id]['images'].append(row['image'])
            # 如果是恶性病变，则设置为True
            lesion_groups[lesion_id]['is_malignant'] = lesion_groups[lesion_id]['is_malignant'] or is_malignant
            # 如果是良性病变，则设置为False
            lesion_groups[lesion_id]['is_benign'] = lesion_groups[lesion_id]['is_benign'] or is_benign

        return lesion_groups

    # 基于病变分组准备样本和权重，实现平衡采样
    def _prepare_samples_and_weights(self):
        """ Prepare samples and weights for balanced sampling. """
        samples = []
        weights = []
        for lesion_id, details in self.lesion_groups.items():
            # 是否属于恶性病变
            is_malignant = details['is_malignant']
            for img_name in details['images']:
                # 包含每个元组的病变id和名字
                samples.append((lesion_id, img_name))
                weights.append(1.0 if is_malignant else 0.5)  # Adjust the weight here for balancing

        return samples, weights

    # 返回数据集的总长度， 应用分组则返回sample长度，否则返回原始数据表的长度
    def __len__(self):
        return len(self.data) if not self.use_groups else len(self.samples)

    # 获取数据样本
    def __getitem__(self, idx):
        # 格式转换
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.use_groups:
            lesion_id, img_name = self.samples[idx]
        else:
            # 从idx行中提取image列的内容
            img_name = self.data.iloc[idx]['image']

        # 拼接图片路径
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        # 转化为RGB模式
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Retrieve the specific disease diagnosis labels
        # 根据图像的名字找到对应的行
        row = self.data[self.data['image'] == img_name].iloc[0]
        # diagnosis_labels = torch.tensor(row[self.labels].values.astype(float))
        # 将提取的标签值转化为float类型的张量
        diagnosis_labels = torch.tensor(row[self.labels].values.astype(float), dtype=torch.float32)

        # 根据是否分组，判断当前样本是否为恶性病变
        if self.use_groups:
            # Get the overall label (benign or malignant) based on the lesion group
            is_malignant = self.lesion_groups[lesion_id]['is_malignant']
        else:
            # Default to no malignancy information if not using groups
            is_malignant = torch.tensor(0.0, dtype=torch.float32)

        # return image, diagnosis_labels, torch.tensor(is_malignant, dtype=torch.float32)
        # torch.tensor 会重新分配一个新的张量，会带来不必要的内存拷贝和梯度追踪问题
        return image, diagnosis_labels, torch.tensor(is_malignant,
                                                     dtype=torch.float32).clone().detach()  # 确保模型输入返回时张量不会有梯度计算的依赖，也不会共享相同的内存空间


def get_transforms(is_train=True):
    if is_train:
        # 训练阶段
        return transforms.Compose([
            transforms.Resize((224, 224)),
            # 以百分之五十的概率随机水平反转图像，增加图像的多样性
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机旋转[-15, 15]度，进一步增加图像的多样性
            transforms.RandomRotation(degrees=15),
            # 随机改变图像的亮度，对比度，饱和度，色调，模拟不同光照条件下的图像变化
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 随机参见图片的尺寸，[80%， 100%]之间，增加数据的局部多样性
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            # 转为pytorch张量
            transforms.ToTensor(),
            # 使用预定义的均值和标准差对图像进行标准化，使模型在训练时更加稳定
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证和测试阶段
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
