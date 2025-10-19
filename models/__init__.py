from .transformers import ViT 
# from .transformers import TextTransformer 

from .mlp_based import gMLP, ResMLP, MLPMixer

# 方便在 train.py 中通过配置名称获取模型
MODEL_REGISTRY = {
    'ViT': ViT,
    'gMLP': gMLP,
    'ResMLP': ResMLP,
    'MLPMixer': MLPMixer,
}