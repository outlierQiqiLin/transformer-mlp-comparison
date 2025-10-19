import torch
import torch.nn as nn
from typing import Tuple
from .patch_embedding import PatchEmbedding

class SpatialGatingUnit(nn.Module):
    def __init__(self, num_patches, embed_dim, dropout_rate=0.):
        """
        gMLP 的空间门控单元 (SGU)。
        作用于 Token 维度 N。
        """
        super().__init__()
        # SGU 的线性层作用于 N 维度
        self.norm = nn.LayerNorm(embed_dim)
        
        # SGU 核心：作用于 N 维度 (num_patches) 的 MLP，并包含激活函数
        self.proj = nn.Linear(num_patches, num_patches)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, N, H/2)
        
        # 1. LayerNorm (沿 D 维，这里是 H/2 维)
        y = self.norm(x)
        
        # 2. 核心转置：(B, N, H/2) -> (B, H/2, N)
        # 这样 Linear 层才能作用于 N 维度
        y = y.transpose(1, 2)
        
        # 3. 空间投影 (跨 Token 混合)：(B, H/2, N) -> (B, H/2, N)
        y = self.proj(y)
        y = self.act(y)
        y = self.dropout(y)

        # 4. 转置回来：(B, H/2, N) -> (B, N, H/2)
        y = y.transpose(1, 2)
        
        return y


class gMLPBlock(nn.Module):
    def __init__(self, dim, num_patches, expansion_factor=2, dropout_rate=0.):
        """
        一个完整的 gMLP Block。

        Args:
            dim (int): 输入和输出特征的维度 D。
            num_patches (int): 序列长度 N。
            expansion_factor (int): 内部扩展的倍数 (通常是 2)。
        """
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        half_inner_dim = inner_dim // 2
        
        self.norm = nn.LayerNorm(dim)
        
        # 1. 通道扩展（特征提取）：D -> H=2D
        self.fc1 = nn.Linear(dim, inner_dim)
        
        # 2. SGU 门控
        self.sgu = SpatialGatingUnit(num_patches, half_inner_dim, dropout_rate)
        
        # 3. 通道收缩：H/2 -> D
        self.fc2 = nn.Linear(half_inner_dim, dim)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # x 的形状: (B, N, D)
        residual = x
        
        # 1. LayerNorm
        x = self.norm(x)
        
        # 2. 扩展：(B, N, D) -> (B, N, H)
        x = self.fc1(x)
        
        # 3. 拆分：(B, N, H) -> Z1 (B, N, H/2) 和 Z2 (B, N, H/2)
        Z1, Z2 = x.chunk(2, dim=-1)
        
        # 4. 空间门控：Z2 经过 SGU 处理
        Z2_projected = self.sgu(Z2)
        
        # 5. 门控操作（元素级乘法）：
        gated_output = Z1 * Z2_projected # (B, N, H/2)
        
        # 6. 收缩：(B, N, H/2) -> (B, N, D)
        x = self.fc2(gated_output)
        x = self.dropout(x)
        
        # 7. 残差连接
        return x + residual
    
class gMLP(nn.Module):
    def __init__(self, 
                 img_size, patch_size, in_channels, num_classes, 
                 dim, depth, 
                 expansion_factor=2, dropout_rate=0.):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim, img_size)
        num_patches = self.patch_embed.num_patches
        
        # 1. gMLP Block 堆叠
        self.blocks = nn.ModuleList([
            gMLPBlock(
                dim=dim,
                num_patches=num_patches, # Block 需要 N 维度信息
                expansion_factor=expansion_factor,
                dropout_rate=dropout_rate
            )
            for _ in range(depth)
        ])
        
        # 2. 最终的 LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        # 3. 分类头
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) 
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = x.mean(dim=1) # 全局平均池化
        
        return self.head(x)