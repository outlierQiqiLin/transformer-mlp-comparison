# evaluate.py (通常会独立，但为了简洁，我们写一个函数)

import torch
from utils.profiler import measure_latency
import torch.nn.functional as F

def evaluate(model, eval_loader, device, config):
    """
    评估模型在测试集上的准确率，并测量推理延迟。
    """
    model.eval()
    correct = 0
    total = 0
    
    # 1. 准确率评估
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n评估准确率: {accuracy:.2f}%")

    # 2. 推理延迟测量
    # 创建一个模拟的输入批次进行延迟测量
    # 假设第一个批次的形状是正确的
    dummy_input = next(iter(eval_loader))[0][0].unsqueeze(0) 
    
    avg_latency_ms = measure_latency(model, dummy_input, device)
    
    print(f"平均推理延迟 (ms): {avg_latency_ms:.4f}")
    
    return accuracy, avg_latency_ms