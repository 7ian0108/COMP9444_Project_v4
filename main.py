import torch
import logging
from train import train_model
from test import test_model
from config import setup_train_dirs, setup_test_dirs
from utils import setup_logger

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置训练输出目录和日志
    output_dir, model_dir = setup_train_dirs()
    setup_logger('main', output_dir)
    logging.info("Starting main script...")

    try:
        # 训练模型
        logging.info("Starting training process...")
        train_model(device, output_dir, model_dir)

        # 设置测试目录
        output_dir, model_dir = setup_test_dirs()

        # 测试模型
        logging.info("Starting testing process...")
        test_model(device, model_dir, output_dir)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    logging.info("Main script completed.")


if __name__ == "__main__":
    main()
