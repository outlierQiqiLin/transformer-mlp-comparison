import yaml
import os
from collections import namedtuple

# 推荐使用 namedtuple 或字典，这里使用字典更灵活

def load_config(config_path):
    """
    加载并解析 YAML 配置文件。

    Args:
        config_path (str): YAML 配置文件的完整路径。

    Returns:
        dict: 包含所有配置参数的字典。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            raise

    # 简单检查一些关键参数
    if 'model_name' not in config or 'dataset_name' not in config:
        raise ValueError("Config file must specify 'model_name' and 'dataset_name'")

    return config

# 示例使用 (在你的 train.py 中):
# from utils.config_loader import load_config
# config_file = 'config/cifar10_vit_base.yaml'
# args = load_config(config_file)
# print(args['learning_rate'])  # 输出 3e-4