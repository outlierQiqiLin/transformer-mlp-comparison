import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        """
        ViT 的 Patch Embedding 层。
        
        Args:
            in_channels (int): 输入通道数 (CIFAR-10 为 3)。
            patch_size (int): 图像块边长 (例如 4)。
            embed_dim (int): 投影到的隐藏维度 (D)。
            img_size (int): 图像边长 (CIFAR-10 为 32)。
        """
        super().__init__()
        
        # 1. 使用 nn.Conv2d 实现 Patching 和 Linear Projection
        # 卷积核大小 = 步长 = patch_size，这确保了不重叠的分块
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 2. 计算输出的 Patch 数量 N
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x 的形状: (Batch_Size, C, H, W)
        
        # 1. 卷积：(B, C, H, W) -> (B, Embed_Dim, N_h, N_w)
        x = self.proj(x)
        
        # 2. 展平 (Flatten): (B, D, N_h, N_w) -> (B, D, N)
        # 也就是将空间维度 (N_h x N_w) 展平为 N
        x = x.flatten(2) 
        
        # 3. 转置 (Transpose): (B, D, N) -> (B, N, D)
        # Transformer 期望的输入形状是 (Batch_Size, Sequence_Length, Feature_Dimension)
        x = x.transpose(1, 2) 
        
        return x