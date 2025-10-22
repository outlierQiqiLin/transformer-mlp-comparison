# train_resmlp_ablation.py
"""
消融实验训练脚本
用法:
    python train_resmlp_ablation.py --config config/cifar10_resmlp_ablation.yaml --variant baseline
    python train_resmlp_ablation.py --config config/cifar10_resmlp_ablation.yaml --variant attn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time

from utils.config import load_config, save_config
from models.ablation.resmlp_ablation import ResMLP_Ablation


def create_ablation_model(config, variant: str, device):
    """
    为消融实验创建模型
    
    Args:
        config: 配置字典
        variant: 模型变体名称 (baseline, attn, no_affine, etc.)
        device: 设备
    """
    # 构造模型配置的 key
    model_key = f'resmlp_{variant}'
    model_config = config.get(f'models.{model_key}')
    
    if model_config is None:
        raise ValueError(f"配置文件中没有找到模型 '{model_key}' 的配置")
    
    num_classes = config.get('dataset.num_classes')
    
    # 创建消融模型
    model = ResMLP_Ablation(
        img_size=config.get('dataset.img_size'),
        patch_size=model_config['patch_size'],
        in_channels=config.get('dataset.in_channels', 3),
        dim=model_config['dim'],
        depth=model_config['depth'],
        layerscale_init=model_config.get('layerscale_init', 0.1),
        num_classes=num_classes,
        expansion_factor=model_config.get('expansion_factor', 4),
        dropout_rate=model_config.get('dropout_rate', 0.0),
        num_heads=model_config.get('num_heads', 8),
        variant=variant  # 关键参数：指定变体
    )
    
    return model.to(device)


def get_dataloader(config):
    """创建数据加载器（与原训练脚本相同）"""
    dataset_name = config.get('dataset.name')
    img_size = config.get('dataset.img_size')
    batch_size = config.get('training.batch_size')
    num_workers = config.get('training.num_workers', 4)
    
    if dataset_name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        try:
            train_dataset = datasets.CIFAR10(
                root='./data', 
                train=True, 
                download=True, 
                transform=train_transform
            )
            
            test_dataset = datasets.CIFAR10(
                root='./data', 
                train=False, 
                download=True, 
                transform=test_transform
            )
        except Exception as e:
            print(f"\n❌ 自动下载失败: {str(e)}")
            print("请先手动下载数据集")
            raise
        
    elif dataset_name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        try:
            train_dataset = datasets.CIFAR100(
                root='./data', 
                train=True, 
                download=True, 
                transform=train_transform
            )
            
            test_dataset = datasets.CIFAR100(
                root='./data', 
                train=False, 
                download=True, 
                transform=test_transform
            )
        except Exception as e:
            print(f"\n❌ 自动下载失败")
            raise
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def create_optimizer(model, config):
    """创建优化器"""
    optimizer_config = config.get('training.optimizer')
    optimizer_name = optimizer_config['name']
    lr = float(optimizer_config['lr'])
    weight_decay = float(optimizer_config.get('weight_decay', 0))
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """创建学习率调度器"""
    scheduler_config = config.get('training.scheduler')
    
    if scheduler_config is None:
        return None
    
    scheduler_name = scheduler_config['name']
    
    if scheduler_name == 'cosine':
        epochs = config.get('training.epochs')
        min_lr = scheduler_config.get('min_lr', 0)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=min_lr
        )
    elif scheduler_name == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
    elif scheduler_name == 'multistep':
        milestones = scheduler_config.get('milestones', [60, 120, 160])
        gamma = scheduler_config.get('gamma', 0.2)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=milestones, 
            gamma=gamma
        )
    else:
        return None
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """验证模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Validating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, save_path)


def train(model, train_loader, test_loader, config, save_dir, device):
    """完整的训练流程"""
    epochs = config.get('training.epochs')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f"\n{'='*60}")
    print(f"开始训练...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'\nEpoch: {epoch}/{epochs}')
        print(f'Time: {epoch_time:.2f}s | LR: {current_lr:.6f}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            print(f'✅ 验证准确率提升: {best_acc:.2f}% -> {test_acc:.2f}%')
            best_acc = test_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(save_dir, 'best_model.pth')
            )
        
        if epoch % config.get('logging.save_freq', 50) == 0:
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        print('-' * 60)
    
    save_checkpoint(
        model, optimizer, epochs, best_acc,
        os.path.join(save_dir, 'final_model.pth')
    )
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存路径: {save_dir}")
    print(f"{'='*60}\n")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_acc': best_acc
    }


def main():
    parser = argparse.ArgumentParser(description='ResMLP 消融实验训练')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['baseline', 'attn', 'no_affine', 
                               'no_layerscale', 'no_cross_patch', 'full'],
                       help='模型变体')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    args = parser.parse_args()
    
    # 1. 加载配置文件
    print(f"📂 加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 2. 显示关键配置
    variant_names = {
        'baseline': 'ResMLP (Baseline)',
        'attn': 'ResMLP + Attention',
        'no_affine': 'ResMLP + LayerNorm',
        'no_layerscale': 'ResMLP - LayerScale',
        'no_cross_patch': 'ResMLP - CrossPatch',
        'full': 'ResMLP Full (类 ViT)'
    }
    
    print("\n" + "="*60)
    print("消融实验配置:")
    print(f"  变体: {variant_names[args.variant]}")
    print(f"  数据集: {config.get('dataset.name')}")
    print(f"  批量大小: {config.get('training.batch_size')}")
    print(f"  训练轮数: {config.get('training.epochs')}")
    print(f"  学习率: {config.get('training.optimizer.lr')}")
    print(f"  优化器: {config.get('training.optimizer.name')}")
    if config.get('training.scheduler'):
        print(f"  调度器: {config.get('training.scheduler.name')}")
    print("="*60 + "\n")
    
    # 3. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 4. 创建保存目录
    exp_name = f"resmlp_{args.variant}_{config.get('dataset.name')}"
    save_dir = os.path.join(config.get('logging.checkpoint_dir'), exp_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 5. 保存配置
    save_config(config, os.path.join(save_dir, 'config.yaml'))
    print(f"✅ 配置已保存到: {os.path.join(save_dir, 'config.yaml')}")
    
    # 6. 创建模型
    model = create_ablation_model(config, args.variant, device)
    
    param_count = model.count_parameters()
    
    print(f"✅ 模型创建成功")
    print(f"   变体: {args.variant}")
    print(f"   参数量: {param_count:,}")
    print(f"   保存目录: {save_dir}\n")
    
    # 7. 加载数据
    print("📊 加载数据集...")
    train_loader, test_loader = get_dataloader(config)
    print(f"   训练样本: {len(train_loader.dataset)}")
    print(f"   测试样本: {len(test_loader.dataset)}")
    print(f"   批次数: {len(train_loader)} (train), {len(test_loader)} (test)\n")
    
    # 8. 从检查点恢复
    if args.resume:
        print(f"📥 从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   从 epoch {checkpoint['epoch']} 继续训练\n")
    
    # 9. 开始训练
    results = train(model, train_loader, test_loader, config, save_dir, device)
    
    # 10. 保存训练结果
    torch.save(results, os.path.join(save_dir, 'training_results.pth'))
    print(f"💾 训练结果已保存到: {os.path.join(save_dir, 'training_results.pth')}")


if __name__ == '__main__':
    main()