import torch.nn as nn

class BasicMLP(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout_rate=0.):
        """
        一个基础的多层感知机块（用于 FFN 或 MLP 架构的核心组件）。

        Args:
            dim (int): 输入和输出特征的维度 (Hidden Dimension, D)。
            expansion_factor (int): 内部隐藏层维度相对于 dim 的扩展倍数。
            dropout_rate (float): Dropout 率。
        """
        super().__init__()
        
        # 内部扩展后的维度
        inner_dim = int(dim * expansion_factor)

        self.net = nn.Sequential(
            # 第一层：从 dim 扩展到 inner_dim
            nn.Linear(dim, inner_dim),
            nn.GELU(), # 使用 GELU 激活函数，这是现代 Transformer/MLP 架构的标准选择
            nn.Dropout(dropout_rate),
            # 第二层：从 inner_dim 恢复到 dim
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)