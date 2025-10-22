# models/ablation/resmlp_ablation.py 
"""
ResMLP 消融实验变体模型 - 用于文本

实验变体：
1. ResMLP (baseline) - 原始模型
2. ResMLP-Attn - 将 cross-patch linear 替换为 attention
3. ResMLP-NoAffine - 使用 LayerNorm 替代 Affine
4. ResMLP-NoLayerScale - 移除 LayerScale
5. ResMLP-NoCrossPatch - 移除 cross-patch sublayer
6. ResMLP-Full - 同时使用 Attention + LayerNorm (接近 ViT)
"""

import torch
import torch.nn as nn
from models.mlp_based.basic_mlp import BasicMLP 
# from models.mlp_based.patch_embedding import PatchEmbedding


class AffineNormalization(nn.Module):
    """Affine 归一化：只做缩放和平移，不计算统计量"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.gamma + self.beta


class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, num_heads=8, dropout_rate=0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output


class ResMLP_Block(nn.Module):
    """原始 ResMLP Block (baseline)"""
    def __init__(self, seq_len, dim, layerscale_init, expansion_factor=4, dropout_rate=0.):
        super().__init__()
        self.affine_1 = AffineNormalization(dim)
        self.affine_2 = AffineNormalization(dim)
        self.linear_patches = nn.Linear(seq_len, seq_len)
        self.mlp_channels = BasicMLP(dim, expansion_factor, dropout_rate)
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones(dim))
    
    def forward(self, x):
        # Cross-patch sublayer
        res_1 = self.affine_1(x)
        res_1 = res_1.transpose(1, 2)
        res_1 = self.linear_patches(res_1)
        res_1 = res_1.transpose(1, 2)
        x = x + self.layerscale_1 * res_1
        
        # Cross-channel sublayer
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + self.layerscale_2 * res_2
        
        return x


class ResMLP_Block_Attn(nn.Module):
    """变体1: 使用 Attention 替代 cross-patch linear"""
    def __init__(self, seq_len, dim, layerscale_init, expansion_factor=4, 
                 dropout_rate=0., num_heads=8):
        super().__init__()
        self.affine_1 = AffineNormalization(dim)
        self.affine_2 = AffineNormalization(dim)
        # ✨ 关键修改：使用 Attention 替代 linear
        self.attention = Attention(dim, num_heads, dropout_rate)
        self.mlp_channels = BasicMLP(dim, expansion_factor, dropout_rate)
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones(dim))
    
    def forward(self, x):
        # Cross-patch sublayer with Attention
        res_1 = self.attention(self.affine_1(x))
        x = x + self.layerscale_1 * res_1
        
        # Cross-channel sublayer
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + self.layerscale_2 * res_2
        
        return x


class ResMLP_Block_NoAffine(nn.Module):
    """变体2: 使用 LayerNorm 替代 Affine"""
    def __init__(self, seq_len, dim, layerscale_init, expansion_factor=4, dropout_rate=0.):
        super().__init__()
        # ✨ 关键修改：使用 LayerNorm
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.linear_patches = nn.Linear(seq_len, seq_len)
        self.mlp_channels = BasicMLP(dim, expansion_factor, dropout_rate)
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones(dim))
    
    def forward(self, x):
        # Cross-patch sublayer
        res_1 = self.norm_1(x)
        res_1 = res_1.transpose(1, 2)
        res_1 = self.linear_patches(res_1)
        res_1 = res_1.transpose(1, 2)
        x = x + self.layerscale_1 * res_1
        
        # Cross-channel sublayer
        res_2 = self.mlp_channels(self.norm_2(x))
        x = x + self.layerscale_2 * res_2
        
        return x


class ResMLP_Block_NoLayerScale(nn.Module):
    """变体3: 移除 LayerScale"""
    def __init__(self, seq_len, dim, layerscale_init, expansion_factor=4, dropout_rate=0.):
        super().__init__()
        self.affine_1 = AffineNormalization(dim)
        self.affine_2 = AffineNormalization(dim)
        self.linear_patches = nn.Linear(seq_len, seq_len)
        self.mlp_channels = BasicMLP(dim, expansion_factor, dropout_rate)
        # ✨ 关键修改：移除 LayerScale
    
    def forward(self, x):
        # Cross-patch sublayer (无 LayerScale)
        res_1 = self.affine_1(x)
        res_1 = res_1.transpose(1, 2)
        res_1 = self.linear_patches(res_1)
        res_1 = res_1.transpose(1, 2)
        x = x + res_1
        
        # Cross-channel sublayer (无 LayerScale)
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + res_2
        
        return x


class ResMLP_Block_NoCrossPatch(nn.Module):
    """变体4: 移除 cross-patch sublayer"""
    def __init__(self, seq_len, dim, layerscale_init, expansion_factor=4, dropout_rate=0.):
        super().__init__()
        # ✨ 关键修改：只保留一个 Affine 和 MLP
        self.affine = AffineNormalization(dim)
        self.mlp_channels = BasicMLP(dim, expansion_factor, dropout_rate)
        self.layerscale = nn.Parameter(layerscale_init * torch.ones(dim))
    
    def forward(self, x):
        # 只有 cross-channel sublayer
        res = self.mlp_channels(self.affine(x))
        x = x + self.layerscale * res
        return x


class ResMLP_Block_Full(nn.Module):
    """变体5: Attention + LayerNorm (最接近 ViT)"""
    def __init__(self, seq_len, dim, layerscale_init, expansion_factor=4, 
                 dropout_rate=0., num_heads=8):
        super().__init__()
        # ✨ 关键修改：同时使用 Attention 和 LayerNorm
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.attention = Attention(dim, num_heads, dropout_rate)
        self.mlp_channels = BasicMLP(dim, expansion_factor, dropout_rate)
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones(dim))
    
    def forward(self, x):
        # Cross-patch sublayer with Attention + LayerNorm
        res_1 = self.attention(self.norm_1(x))
        x = x + self.layerscale_1 * res_1
        
        # Cross-channel sublayer with LayerNorm
        res_2 = self.mlp_channels(self.norm_2(x))
        x = x + self.layerscale_2 * res_2
        
        return x


# class Patch_projector(nn.Module):
#     """Patch 投影层"""
#     def __init__(self, in_channels=3, patch_size=16, dim=384):
#         super().__init__()
#         self.patch_size = patch_size
#         self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.projection(x)
#         x = x.flatten(2)
#         x = x.transpose(1, 2)
#         return x

class TextEmbedding(nn.Module):
    """
    文本嵌入层 (替换 Patch_projector)
    包含 Token 嵌入和 Positional 嵌入
    """
    def __init__(self, vocab_size, dim, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        # 使用可学习的位置嵌入
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        # input_ids: [B, S]
        B, S = input_ids.shape
        
        # 1. Token 嵌入
        x_token = self.token_embedding(input_ids) # [B, S, D]
        
        # 2. 位置嵌入
        # [1, S, D] (截取或填充到当前序列长度 S)
        # 注意：我们只取 'S' 长度的位置编码，以匹配 input_ids
        x_pos = self.position_embedding[:, :S, :] 
        
        # 3. 相加
        x = x_token + x_pos
        x = self.dropout(x)
        
        return x


class ResMLP_Ablation_Text(nn.Module):
    """
    ResMLP 消融实验模型
    
    Args:
        variant: 模型变体
            - 'baseline': 原始 ResMLP
            - 'attn': 使用 Attention 替代 linear
            - 'no_affine': 使用 LayerNorm 替代 Affine
            - 'no_layerscale': 移除 LayerScale
            - 'no_cross_patch': 移除 cross-patch sublayer
            - 'full': Attention + LayerNorm (类 ViT)
    """
    def __init__(self, 
                 vocab_size,     # (新增) 词汇表大小
                 max_seq_len,    # (新增) 最大序列长度
                 dim=384, 
                 depth=12, 
                 layerscale_init=0.1, 
                 num_classes=2,  # (修改) 默认为2 (SST-2)
                 expansion_factor=4,
                 dropout_rate=0.,
                 num_heads=8,
                 variant='baseline'):
        super().__init__()
        
        self.variant = variant
        
        self.embedding = TextEmbedding(
            vocab_size=vocab_size, 
            dim=dim, 
            max_seq_len=max_seq_len, 
            dropout_rate=dropout_rate
        )
        
        # 根据变体选择 Block 类型
        if variant == 'baseline':
            block_class = ResMLP_Block
            block_kwargs = {}
        elif variant == 'attn':
            block_class = ResMLP_Block_Attn
            block_kwargs = {'num_heads': num_heads}
        elif variant == 'no_affine':
            block_class = ResMLP_Block_NoAffine
            block_kwargs = {}
        elif variant == 'no_layerscale':
            block_class = ResMLP_Block_NoLayerScale
            block_kwargs = {}
        elif variant == 'no_cross_patch':
            block_class = ResMLP_Block_NoCrossPatch
            block_kwargs = {}
        elif variant == 'full':
            block_class = ResMLP_Block_Full
            block_kwargs = {'num_heads': num_heads}
        else:
            raise ValueError(f"未知变体: {variant}")
        
        # 构建 blocks
        self.blocks = nn.ModuleList([
            block_class(
                seq_len=max_seq_len,
                dim=dim,
                layerscale_init=layerscale_init,
                expansion_factor=expansion_factor,
                dropout_rate=dropout_rate,
                **block_kwargs
            )
            for _ in range(depth)
        ])
        
        # Final normalization
        if variant in ['no_affine', 'full']:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = AffineNormalization(dim)
        
        # Classification head
        self.head = nn.Linear(dim, num_classes)

    def forward(self, input_ids):
        # input_ids: [B, S]
        x = self.embedding(input_ids) # [B, S, D]
        
        # Apply blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.head(x)
        
        return x
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 便捷函数
def resmlp_baseline(**kwargs):
    """原始 ResMLP"""
    return ResMLP_Ablation_Text(variant='baseline', **kwargs)


def resmlp_with_attention(**kwargs):
    """ResMLP + Attention"""
    return ResMLP_Ablation_Text(variant='attn', **kwargs)


def resmlp_with_layernorm(**kwargs):
    """ResMLP + LayerNorm"""
    return ResMLP_Ablation_Text(variant='no_affine', **kwargs)


def resmlp_no_layerscale(**kwargs):
    """ResMLP without LayerScale"""
    return ResMLP_Ablation_Text(variant='no_layerscale', **kwargs)


def resmlp_no_cross_patch(**kwargs):
    """ResMLP without cross-patch sublayer"""
    return ResMLP_Ablation_Text(variant='no_cross_patch', **kwargs)


def resmlp_vit_like(**kwargs):
    """ResMLP with Attention + LayerNorm (类 ViT)"""
    return ResMLP_Ablation_Text(variant='full', **kwargs)