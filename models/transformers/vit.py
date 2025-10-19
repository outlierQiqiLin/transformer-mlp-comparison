import torch
import torch.nn as nn
from models.mlp_based.basic_mlp import BasicMLP 
from models.mlp_based.patch_embedding import PatchEmbedding

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout_rate=0.):
        super().__init__()
        # 使用 PyTorch 内置的 MultiheadAttention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            batch_first=True  # 确保输入是 (B, N, D)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 注意 PyTorch 的 MultiheadAttention 需要 Q, K, V
        # 对于自注意力，Q=K=V=x
        # output 是一个元组 (attn_output, attn_output_weights)
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion_factor, dropout_rate=0.):
        super().__init__()
        
        # 1. MSA 子层 (带残差连接和 LayerNorm)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout_rate)
        
        # 2. FFN 子层 (带残差连接和 LayerNorm)
        self.norm2 = nn.LayerNorm(dim)
        # FFN 使用 BasicMLP
        self.mlp = BasicMLP(dim, expansion_factor, dropout_rate)

    def forward(self, x):
        # 第一个残差连接：MSA
        x = x + self.attn(self.norm1(x))
        
        # 第二个残差连接：FFN
        x = x + self.mlp(self.norm2(x))
        
        return x
    

class ViT(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 num_classes=1000, 
                 dim=768,  # embed_dim 的别名
                 depth=12, 
                 num_heads=12, 
                 mlp_expansion_factor=4, 
                 dropout_rate=0.):
        """
        Vision Transformer (ViT)
        
        Args:
            img_size: 输入图像大小
            patch_size: patch 大小
            in_channels: 输入通道数
            num_classes: 分类数量
            dim: embedding 维度（也叫 embed_dim）
            depth: Transformer 层数
            num_heads: 注意力头数
            mlp_expansion_factor: MLP 隐藏层扩展因子（也叫 mlp_ratio）
            dropout_rate: dropout 比率
        """
        super().__init__()

        # 1. Patch Projection
        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim, img_size)
        num_patches = self.patch_embed.num_patches
        
        # 2. [CLS] Token (可学习的，用于分类)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # 3. 位置编码 (Positional Encoding)
        # 序列长度是 num_patches + 1 (for CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        
        # 4. Dropout (应用于 Positional Embedding)
        self.pos_drop = nn.Dropout(p=dropout_rate)
        
        # 5. Transformer Encoder Block 堆叠
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim, num_heads, mlp_expansion_factor, dropout_rate
            )
            for _ in range(depth)
        ])
        
        # 6. 最终的 LayerNorm 和分类头
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # 初始化 Positional Embedding (通常使用截断正态分布)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patching: (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x) 
        
        # 2. 添加 [CLS] Token
        # 扩展 [CLS] token 以匹配 Batch Size: (1, 1, D) -> (B, 1, D)
        cls_token = self.cls_token.expand(B, -1, -1)
        # 拼接：(B, 1, D) + (B, N, D) -> (B, N+1, D)
        x = torch.cat((cls_token, x), dim=1) 
        
        # 3. 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. 经过 Transformer Blocks
        for blk in self.blocks:
            x = blk(x)

        # 5. 归一化，并只取出 [CLS] Token 的输出
        x = self.norm(x)
        cls_token_output = x[:, 0]  # 取出索引 0 上的 [CLS] token
        
        # 6. 分类
        return self.head(cls_token_output)
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)