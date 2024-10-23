import os
from datetime import datetime

# 定义项目的基础目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据集路径（相对路径）存储在基础目录下的skin_lesion/dataset 文件夹中
dataset_dir = os.path.join(base_dir, 'skin_lesion', 'dataset')
# 其他配置参数
NUM_CLASSES = 7
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
NUM_EPOCHS = 50
# 定义早停法的耐心值
EARLY_STOPPING_PATIENCE = 9


# 为训练创建新的输出目录
def get_train_output_dir():
    # 每次训练之后，创建一个带有时间戳的新的输出目录
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # 输出的路径
    output_dir = os.path.join('output', timestamp)
    # 创建制定的输出目录
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# 获取最新的输出目录(用于测试)
def get_latest_output_dir():
    output_base = 'output'
    # 检查是否存在 output 目录
    if not os.path.exists(output_base):
        # 没有抛出异常
        raise FileNotFoundError("No output directory found. Please run training first.")

    # 获取一个目录中的所有的子目录路径
    subdirs = [os.path.join(output_base, d) for d in os.listdir(output_base) if
               os.path.isdir(os.path.join(output_base, d))]
    # 如果没有子目录
    if not subdirs:
        # 抛出异常
        raise FileNotFoundError("No output subdirectories found. Please run training first.")

    # 找到最新修改的目录
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir


# 训练时调用此函数， 创建模型保存的路径
def setup_train_dirs():
    output_dir = get_train_output_dir()
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return output_dir, model_dir


# 测试时调用此函数， 创建模型保存的路径
def setup_test_dirs():
    output_dir = get_latest_output_dir()
    model_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Models directory not found in {output_dir}")
    return output_dir, model_dir
