import torch
import torch.nn as nn
from .basic_mlp import BasicMLP 

class AffineNormalization(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # gamma (缩放因子) 初始化为 1
        self.gamma = nn.Parameter(torch.ones(dim))
        # beta (偏移因子) 初始化为 0
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x 的形状是 (B, N, D)
        # AffN 直接对 D 维度进行缩放和偏移，不涉及均值/方差计算
        return x * self.gamma + self.beta

# class Mlp(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, 4 * dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(4 * dim, dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return x

# Patch projection layer to convert image to patches
class Patch_projector(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, dim)
        return x


# ResMLP Block: linear layer on patches + MLP on channels
class ResMLP_Block(nn.Module):
    def __init__(self, nb_patches, dim, layerscale_init, expansion_factor=4, dropout_rate=0.):
        super().__init__()
        # 修复: Affine -> AffineNormalization
        self.affine_1 = AffineNormalization(dim)
        self.affine_2 = AffineNormalization(dim)
        self.linear_patches = nn.Linear(nb_patches, nb_patches)
        
        # 使用 BasicMLP
        self.mlp_channels = BasicMLP(
            dim=dim, 
            expansion_factor=expansion_factor, 
            dropout_rate=dropout_rate
        )
        
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones(dim))
    
    def forward(self, x):
        # First residual: linear layer on patches
        res_1 = self.affine_1(x)
        res_1 = res_1.transpose(1, 2)
        res_1 = self.linear_patches(res_1)
        res_1 = res_1.transpose(1, 2)
        x = x + self.layerscale_1 * res_1
        
        # Second residual: MLP on channels
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + self.layerscale_2 * res_2
        
        return x


# Complete ResMLP model
class ResMLP(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 dim=384, 
                 depth=12, 
                 layerscale_init=0.1, 
                 num_classes=1000,
                 expansion_factor=4,
                 dropout_rate=0.):
        super().__init__()
        
        # Calculate number of patches
        self.nb_patches = (img_size // patch_size) ** 2
        
        # Patch projector
        self.patch_projector = Patch_projector(in_channels, patch_size, dim)
        
        # ResMLP blocks
        self.blocks = nn.ModuleList([
            ResMLP_Block(self.nb_patches, dim, layerscale_init, expansion_factor, dropout_rate)
            for _ in range(depth)
        ])
        
        # Final affine transformation
        # 修复: Affine -> AffineNormalization
        self.affine = AffineNormalization(dim)
        
        # Classification head
        self.linear_classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert image to patches
        x = self.patch_projector(x)  # (B, nb_patches, dim)
        
        # Apply ResMLP blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final affine transformation
        x = self.affine(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, dim)
        
        # Classification
        x = self.linear_classifier(x)
        
        return x
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)