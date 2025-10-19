import torch
import time
from contextlib import contextmanager

class ModelProfiler:
    """模型效率分析工具"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.train_time = 0
        self.inference_time = 0
        self.peak_memory = 0
    
    @contextmanager
    def profile_training(self):
        """分析训练效率"""
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        yield
        
        self.train_time = time.time() - start_time
        self.peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    def profile_inference(self, dataloader, num_batches=100):
        """分析推理效率"""
        self.model.eval()
        
        total_time = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                else:
                    inputs = batch[0].to(self.device)
                
                # 预热
                if i == 0:
                    if isinstance(inputs, dict):
                        _ = self.model(**inputs)
                    else:
                        _ = self.model(inputs)
                    continue
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                if isinstance(inputs, dict):
                    _ = self.model(**inputs)
                else:
                    _ = self.model(inputs)
                
                torch.cuda.synchronize()
                total_time += time.time() - start_time
                total_samples += inputs['input_ids'].size(0) if isinstance(inputs, dict) else inputs.size(0)
        
        avg_latency = (total_time / (num_batches - 1)) * 1000  # ms
        throughput = total_samples / total_time
        
        return {
            'latency_ms': avg_latency,
            'throughput': throughput
        }
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'parameters': self.model.count_parameters(),
            'train_time_sec': self.train_time,
            'peak_memory_mb': self.peak_memory
        }