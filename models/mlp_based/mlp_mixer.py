import torch
import torch.nn as nn
# 导入你提供的基础模块
from .basic_mlp import BasicMLP 
from .patch_embedding import PatchEmbedding


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_expansion_factor, channel_expansion_factor, dropout_rate=0.):
        super().__init__()
        
        # === Token-Mixing Sub-Layer (跨 Token N 维度混合) ===
        self.norm1 = nn.LayerNorm(dim)
        
        # Token MLP 的输入和输出维度是 num_patches (N)
        # 注意：这里的 BasicMLP 需要一个特殊的参数处理，
        # 因为它的输入不是 dim，而是 num_patches。
        # 我们在这里直接定义 Linear 层以保持逻辑清晰，或者修改 BasicMLP 使其更通用。
        
        token_hidden_dim = int(num_patches * token_expansion_factor)
        
        # 我们直接使用 Linear 层来处理非 D 维度的操作
        self.token_mlp_net = nn.Sequential(
            nn.Linear(num_patches, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(token_hidden_dim, num_patches),
            nn.Dropout(dropout_rate)
        )
        
        # === Channel-Mixing Sub-Layer (跨特征 D 维度混合) ===
        self.norm2 = nn.LayerNorm(dim)
        
        # Channel MLP 的输入和输出维度是 dim (D)，可以直接使用 BasicMLP
        self.channel_mlp = BasicMLP(
            dim=dim, 
            expansion_factor=channel_expansion_factor, 
            dropout_rate=dropout_rate
        )
        
    def forward(self, x):
        # x 的形状: (B, N, D)
        
        # 1. Token-Mixing Sub-Layer (处理 N 维度):
        y = self.norm1(x)
        y_transposed = y.transpose(1, 2) # (B, N, D) -> (B, D, N)
        
        # Token MLP 作用于最后一个维度 N
        y_mixed = self.token_mlp_net(y_transposed) 
        
        # 转置回来：(B, D, N) -> (B, N, D)
        y = y_mixed.transpose(1, 2) 
        
        # 残差连接
        x = x + y 
        
        # 2. Channel-Mixing Sub-Layer (处理 D 维度):
        y = self.norm2(x)
        
        # Channel MLP 作用于最后一个维度 D
        y = self.channel_mlp(y)
        
        # 残差连接
        x = x + y
        
        return x
    
class MLPMixer(nn.Module):
    def __init__(self, 
                 img_size, patch_size, in_channels, num_classes, 
                 dim, depth, 
                 token_expansion_factor=0.5, channel_expansion_factor=4, 
                 dropout_rate=0.):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim, img_size)
        num_patches = self.patch_embed.num_patches
        
        # 1. Mixer Block 堆叠
        self.blocks = nn.ModuleList([
            MixerBlock(
                dim=dim,
                num_patches=num_patches,
                token_expansion_factor=token_expansion_factor,
                channel_expansion_factor=channel_expansion_factor,
                dropout_rate=dropout_rate
            )
            for _ in range(depth)
        ])
        
        # 2. 最终的 LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        # 3. 分类头 (不需要 CLS token)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # 1. Patching: (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x) 
        
        # 2. 经过 Mixer Blocks
        for blk in self.blocks:
            x = blk(x)
        
        # 3. 归一化
        x = self.norm(x)
        
        # 4. 全局平均池化 (Global Average Pooling): (B, N, D) -> (B, D)
        # 这是 MLP-based 模型获取全局信息的方式
        x = x.mean(dim=1) 
        
        # 5. 分类
        return self.head(x)