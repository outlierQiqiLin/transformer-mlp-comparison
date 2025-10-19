# data/cifar10_loader.py

import torchvision.transforms as T
import torchvision.datasets as datasets # <-- 新增导入
from torch.utils.data import DataLoader

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2471, 0.2435, 0.2616]

# 定义 transforms（保持不变）
# ... train_transforms 和 test_transforms 的定义 ...

def load_cifar10_data(batch_size, num_workers=0):
    # 训练集的数据增强和预处理
    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # 测试集/验证集
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # === 关键修正：实例化 Dataset 对象 ===
    train_dataset = datasets.CIFAR10(
        root='./data_cache',   # 数据存放路径
        train=True,            # 加载训练集
        download=True,         # 如果没有，则下载
        transform=train_transforms # 应用训练集转换
    )
    
    eval_dataset = datasets.CIFAR10(
        root='./data_cache',
        train=False,           # 加载测试集 (作为评估集)
        download=True,
        transform=test_transforms # 应用测试集转换
    )
    
    # 使用 Dataset 对象初始化 DataLoader
    train_dataloader = DataLoader(
        train_dataset,  
        shuffle=True, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    eval_dataloader = DataLoader(
        eval_dataset,  
        shuffle=False, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_dataloader, eval_dataloader