import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    """配置管理类 - 只负责加载和访问配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default=None):
        """
        获取配置值，支持点号访问嵌套字典
        例如: config.get('models.vit.embed_dim')
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """返回完整配置字典"""
        return self._config
    
    def __repr__(self):
        return f"Config({yaml.dump(self._config, default_flow_style=False, indent=2)})"


def load_config(config_path: str) -> Config:
    """从YAML文件加载配置"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """保存配置到YAML文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    print(f"✅ 配置已保存到: {save_path}")