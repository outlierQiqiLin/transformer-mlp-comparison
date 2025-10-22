# evaluate_efficiency.py
"""
评估模型效率 (Latency 和 Throughput)

用法:
    python evaluate_efficiency.py \
        --config ./results/ablation_study/resmlp_baseline_cifar10/config.yaml \
        --variant baseline \
        --checkpoint ./results/ablation_study/resmlp_baseline_cifar10/best_model.pth \
        --batch_size 64
"""

import torch
import time
import argparse
from tqdm import tqdm
import numpy as np

from utils.config import load_config
from models.ablation.resmlp_ablation import ResMLP_Ablation
from train_resmlp_ablation import create_ablation_model

def measure_efficiency(model, device, img_size, batch_size=64, warmup_steps=50, measure_steps=200):
    """
    测量模型的 Latency 和 Throughput
    """
    model.eval()
    
    # 1. 测量 Latency (单样本延迟)
    print(f"\n--- 1. 测量 Latency (Batch Size = 1) ---")
    dummy_input_latency = torch.randn(1, 3, img_size, img_size).to(device)
    
    # 预热
    print(f"预热 (Warmup) {warmup_steps} 步...")
    for _ in range(warmup_steps):
        _ = model(dummy_input_latency)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    print(f"测量 {measure_steps} 步...")
    timings = []
    with torch.no_grad():
        for _ in tqdm(range(measure_steps)):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = model(dummy_input_latency)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            
    avg_latency_s = np.mean(timings)
    avg_latency_ms = avg_latency_s * 1000
    
    print(f"✅ 平均 Latency: {avg_latency_ms:.3f} ms")

    # 2. 测量 Throughput (指定 Batch Size 吞吐量)
    print(f"\n--- 2. 测量 Throughput (Batch Size = {batch_size}) ---")
    dummy_input_throughput = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # 预热
    print(f"预热 (Warmup) {warmup_steps} 步...")
    for _ in range(warmup_steps):
        _ = model(dummy_input_throughput)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"测量 {measure_steps} 步...")
    total_time = 0.0
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_total_time = time.perf_counter()
        
        for _ in tqdm(range(measure_steps)):
            _ = model(dummy_input_throughput)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_total_time = time.perf_counter()
        
    total_time = end_total_time - start_total_time
    total_samples = measure_steps * batch_size
    throughput_sps = total_samples / total_time # samples per second
    
    print(f"总样本数: {total_samples}")
    print(f"总时间: {total_time:.3f} s")
    print(f"✅ 平均 Throughput: {throughput_sps:.2f} samples/sec")
    
    return avg_latency_ms, throughput_sps


def main():
    parser = argparse.ArgumentParser(description='ResMLP 效率评估')
    parser.add_argument('--config', type=str, required=True,
                       help='训练时使用的配置文件路径')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['baseline', 'attn', 'no_affine', 
                               'no_layerscale', 'no_cross_patch', 'full'],
                       help='模型变体')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='训练好的模型检查点路径 (e.g., best_model.pth)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='用于测量吞吐量的批量大小')
    args = parser.parse_args()

    # 1. 加载配置
    print(f"📂 加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 3. 创建模型
    model = create_ablation_model(config, args.variant, device)
    
    # 4. 加载权重
    print(f"📥 加载模型权重: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # 兼容性处理：检查点可能保存了 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    # 5. 获取图像大小
    img_size = config.get('dataset.img_size')
    
    # 6. 运行效率测试
    measure_efficiency(
        model, 
        device, 
        img_size, 
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()


